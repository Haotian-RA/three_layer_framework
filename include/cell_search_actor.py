from actor_system import PipelinedActor
from msg_type import CellFound, DataReady, BufferRelease, NoCell



class CellSearchActor(PipelinedActor):
    """Initial search -> tracking on detection. Safe slot resize."""

    def __init__(self, system, buf, pool, graph, params, node_pipeline):
        super().__init__('cell_search', system, pool, graph, node_pipeline[0])
        self._buf = buf
        self._mode = 'initial_search'
        self._track_size = params.pss_tracking_len

    def _default_behavior(self, msg):
        if isinstance(msg, DataReady):
            if self._mode == 'tracking' and msg.tag.get('mode') == 'initial_search':
                return
        super()._default_behavior(msg)

    def _fill_slot(self, slot_idx, msg):
        slot = self._pool.slots[slot_idx]
        data = self._buf.read_protected(msg.pid)
        slot.data[:len(data)] = data
        slot.tag = msg.tag
        self.system.send_message('buffer_manager', BufferRelease(pid=msg.pid))

    def on_result(self, msg):
        if self._mode == 'tracking' and msg.tag.get('mode') == 'initial_search':
            return
        slot = self._pool.slots[msg.slot]
        if slot.sss_detected:
            self.system.send_message('controller', CellFound(
                N_id=3 * slot.N_id_1 + slot.N_id_2,
                f_d=slot.f_d, F=slot.F,
                pss_local_index=slot.pss_local_index,
                tag=msg.tag))
            if self._mode == 'initial_search':
                self._switch_to_tracking()
        else:
            self.system.send_message('controller', NoCell(tag=msg.tag))

    def _switch_to_tracking(self):
        self._mode = 'tracking'
        self._queue.clear()
        self._pool.set_size(self._track_size)
