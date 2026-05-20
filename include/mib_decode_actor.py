from actor_system import PipelinedActor
from msg_type import PipelineDone, BufferRelease, MIBDecoded, Token, MIBFail, Config



class MIBDecodeActor(PipelinedActor):

    ANT_TRIALS = [1, 2, 4]

    def __init__(self, system, buf, pool, graph, node_pipeline):
        # node_pipeline = [crs_node, pbch_node, bch_node]
        super().__init__('mib_decode', system, pool, graph, node_pipeline[0])
        self._buf = buf
        self._pipeline = node_pipeline
        self._known_n_ant = None             # cached after first success
        self._trial_slots = {}               # slot_idx → trial_idx (active blind trials)

    # ---- Override: keep slot alive during blind trial ----

    def _default_behavior(self, msg):
        if isinstance(msg, PipelineDone):
            # on_result returns True = retry (keep slot), False = done (release slot)
            keep = self.on_result(msg)
            if not keep:
                self._pool.release(msg.slot)
                self._in_flight -= 1
            self._try_dispatch()
        elif isinstance(msg, Config):
            self.on_config(msg)
        else:
            # DataReady handled by PipelinedActor (queue + dispatch)
            super()._default_behavior(msg)

    # ---- Fill slot from buffer ----

    def _fill_slot(self, slot_idx, msg):
        slot = self._pool.slots[slot_idx]
        data = self._buf.read_protected(msg.pid)
        slot.data[:len(data)] = data
        slot.tag = msg.tag

        # tag carries cell info from cell search: N_id, f_d
        slot.N_id = msg.tag['N_id']
        slot.f_d = msg.tag['f_d']
        self.system.send_message('buffer_manager', BufferRelease(pid=msg.pid))

        if self._known_n_ant:
            # fast path: use cached antenna count
            slot.n_ant = self._known_n_ant
        else:
            # blind trial: start with first hypothesis
            slot.n_ant = self.ANT_TRIALS[0]
            self._trial_slots[slot_idx] = 0

    # ---- Result handling with blind trial ----

    def on_result(self, msg):
        slot = self._pool.slots[msg.slot]

        if slot.mib_decoded:
            # ---- SUCCESS: cache n_ant, report to controller ----
            self._known_n_ant = slot.n_ant
            self._trial_slots.pop(msg.slot, None)
            self.system.send_message('controller', MIBDecoded(
                dl_bw=slot.dl_bw, n_ant=slot.n_ant,
                sfn=slot.sfn, phich_dur=slot.phich_dur,
                phich_res=slot.phich_res, tag=msg.tag))
            return False  # release slot

        elif msg.slot in self._trial_slots:
            # ---- BLIND TRIAL: try next n_ant hypothesis ----
            trial_idx = self._trial_slots[msg.slot] + 1

            if trial_idx < len(self.ANT_TRIALS):
                # more hypotheses to try — re-enter pipeline at CRS
                self._trial_slots[msg.slot] = trial_idx
                slot.n_ant = self.ANT_TRIALS[trial_idx]
                slot.mib_decoded = False
                self._graph.try_put(self._first_node, Token(slot=msg.slot, tag=msg.tag))
                return True  # keep slot — still in use
            else:
                # all hypotheses exhausted
                self._trial_slots.pop(msg.slot, None)
                self.system.send_message('controller', MIBFail(tag=msg.tag))
                return False  # release slot

        else:
            # ---- CACHED n_ant FAILED: restart blind trial ----
            # CRS is still valid — re-enter at CRS with first hypothesis
            self._known_n_ant = None
            slot.n_ant = self.ANT_TRIALS[0]
            slot.mib_decoded = False
            self._trial_slots[msg.slot] = 0
            self._graph.try_put(self._first_node, Token(slot=msg.slot, tag=msg.tag))
            return True  # keep slot — restarting trial

    # ---- Config: propagate to pipeline stages ----

    def on_config(self, msg):
        for node in self._pipeline:
            if node.name in msg.params:
                node.func.config(**msg.params[node.name])
