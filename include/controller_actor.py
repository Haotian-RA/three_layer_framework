
from actor_system import Actor
from msg_type import CellFound, NoCell, MIBDecoded, MIBFail, SIB1Decoded, SIB1Fail, DataExhausted, Config, BufferRead, BufferConsume



class ControllerActor(Actor):
    """LTE protocol state machine.

    States:
      IDLE        → waiting for 'start'
      CELL_SEARCH → blind PSS/SSS scan over half frame
      TRACKING    → sliding window: PSS → MIB → SIB1 per frame

    Dependency chain per frame:
      PSS done → send MIB
      MIB done → if even SFN: send SIB1, else skip
      SIB1 done → cleanup, consume, replenish
    """

    def __init__(self, system, params, max_tracking=5,
                 sib1_target=None, done_event=None):
        super().__init__('controller', system)
        self.params = params

        # cell parameters — learned from cell search + MIB
        self.N_id = None
        self.dl_bw = None
        self.n_ant = None
        self.phich_dur = None
        self.phich_res = None

        # tracking window
        self.TRACKING_LOOKAHEAD = max_tracking
        self.tracking_table = {}
        self.consume_cursor = 0
        self._next_frame_idx = 0
        self._bootstrap_done = False
        self._sib1_configured = False
        self._data_exhausted = False

        # test support
        self._sib1_target = sib1_target
        self._done_event = done_event

        # results + logs
        self.log = []
        self.sib1_results = []

        self.become(self._idle_behavior)

    # ======== Behaviors ========

    def _idle_behavior(self, msg):
        if msg == 'start':
            self.log.append('IDLE → CELL_SEARCH')
            self.become(self._cell_search_behavior)
            self._dispatch_cell_search()

    def _cell_search_behavior(self, msg):
        if isinstance(msg, CellFound):
            self._on_cell_found_initial(msg)
        elif isinstance(msg, NoCell):
            pass

    def _tracking_behavior(self, msg):
        if isinstance(msg, CellFound):
            self._on_pss_done(msg)
        elif isinstance(msg, NoCell):
            self._on_pss_failed(msg)
        elif isinstance(msg, MIBDecoded):
            self._on_mib_done(msg)
        elif isinstance(msg, MIBFail):
            self._on_mib_failed(msg)
        elif isinstance(msg, SIB1Decoded):
            self._on_sib1_done(msg)
        elif isinstance(msg, SIB1Fail):
            self._on_sib1_failed(msg)
        elif isinstance(msg, DataExhausted):
            self.log.append('TRACKING: data exhausted')
            self._data_exhausted = True
            # don't set done — wait for in-flight work to complete
            self._check_done()

    # ======== Cell Search → Tracking transition ========

    def _on_cell_found_initial(self, msg):
        self.N_id = msg.N_id
        pss_global = msg.tag['pos'] + msg.pss_local_index
        f_d = msg.f_d
        F = msg.F
        p = self.params

        # config MIB pipeline (N_id known now)
        self.system.send_message('mib_decode', Config(params={
            'mib_crs':  {'N_id': self.N_id, 'N_rb': 6, 'ns': 1,
                         'n_symbols': 4, 'is_slot_start': True},
            'mib_pbch': {'N_id': self.N_id},
            'mib_bch':  {'N_id': self.N_id}
        }))

        if F == 0:
            slot1_start = pss_global + p.N_FFT
            frame_start = slot1_start - p.N_slot

            self.log.append(
                f'CELL_SEARCH → TRACKING: N_id={self.N_id}, '
                f'pss_global={pss_global}, frame_start={frame_start}')

            # consume buffer up to frame start
            consume_amount = frame_start - self.consume_cursor
            if consume_amount > 0:
                self._do_consume(consume_amount, 'cell_search_align')

            self.become(self._tracking_behavior)

            # first entry: PSS already done
            self._next_frame_idx += 1
            frame_tag = f'frame_{self._next_frame_idx}'
            entry = self._new_entry(frame_tag)
            entry['pss']['status'] = 'done'
            entry['pss']['pos'] = pss_global
            entry['pss']['f_d'] = f_d
            self.tracking_table[frame_tag] = entry

            # PSS done → send MIB
            self._try_send_mib(frame_tag)

        elif F == 1:
            expected_f0_pss = pss_global + p.N_half_frame

            self.log.append(
                f'CELL_SEARCH → TRACKING (F=1): N_id={self.N_id}, '
                f'next F=0 at {expected_f0_pss}')

            self.become(self._tracking_behavior)
            self._dispatch_pss_tracking(expected_f0_pss)

    # ======== PSS tracking ========

    def _on_pss_done(self, msg):
        frame_tag = msg.tag.get('frame_tag')
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return

        # refine PSS position from tracking window
        track_start = entry['pss']['pos'] - 2 * self.params.N_ofdm_sym
        pss_global = track_start + msg.pss_local_index
        entry['pss']['pos'] = pss_global
        entry['pss']['f_d'] = msg.f_d
        entry['pss']['status'] = 'done'

        self.log.append(f'TRACKING: pss_done {frame_tag}, pos={pss_global}')

        self._try_send_mib(frame_tag)

    def _on_pss_failed(self, msg):
        frame_tag = msg.tag.get('frame_tag')
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return

        self.log.append(f'TRACKING: pss_failed {frame_tag}')
        entry['mib']['status'] = 'done'
        entry['sib1']['status'] = 'done'
        self._try_advance_cursor()
        self._try_cleanup()
        self._check_done()

    # ======== MIB ========

    def _try_send_mib(self, frame_tag):
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return
        if entry['pss']['status'] != 'done':
            return
        if entry['mib']['status'] != 'not_sent':
            return

        p = self.params
        mib_offset = entry['pss']['pos'] + p.N_FFT - self.consume_cursor

        self.system.send_message('buffer_manager', BufferRead(
            offset=mib_offset,
            length=p.pbch_len,
            dest='mib_decode',
            tag={'frame_tag': frame_tag,
                 'N_id': self.N_id,
                 'f_d': entry['pss']['f_d']}))

        entry['mib']['status'] = 'pending'
        self.log.append(f'TRACKING: mib_sent {frame_tag}')

    def _on_mib_done(self, msg):
        frame_tag = msg.tag.get('frame_tag')
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return

        # update cell parameters
        self.dl_bw = msg.dl_bw
        self.n_ant = msg.n_ant
        self.phich_dur = msg.phich_dur
        self.phich_res = msg.phich_res
        entry['mib']['sfn'] = msg.sfn
        entry['mib']['status'] = 'done'

        self.log.append(
            f'TRACKING: mib_done {frame_tag}, SFN={msg.sfn}, '
            f'BW={self.dl_bw}, n_ant={self.n_ant}')

        if msg.sfn % 2 == 0:
            # config SIB1 pipeline on first even SFN (dl_bw, n_ant now known)
            if not self._sib1_configured:
                self._sib1_configured = True
                self.system.send_message('sib1_decode', Config(params={
                    'sib1_crs':    {'N_id': self.N_id, 'N_rb': self.dl_bw, 'ns': 10,
                                    'n_symbols': 14, 'is_slot_start': True},
                    'sib1_pcfich': {'N_id': self.N_id, 'N_rb': self.dl_bw, 'ns': 10},
                    'sib1_pdcch':  {'N_id': self.N_id, 'N_rb': self.dl_bw,
                                    'n_ant': self.n_ant, 'phich_res': self.phich_res,
                                    'ns': 10},
                    'sib1_pdsch':  {'N_id': self.N_id, 'n_ant': self.n_ant},
                    'sib1_dlsch':  {'ns': 10, 'N_id': self.N_id}
                }))
            self._try_send_sib1(frame_tag)
        else:
            self.log.append(f'TRACKING: SFN={msg.sfn} odd, skip SIB1')
            entry['sib1']['status'] = 'done'
            self._try_advance_cursor()
            self._try_cleanup()
            self._check_done()

        # bootstrap: fill lookahead after first MIB
        if not self._bootstrap_done:
            self._bootstrap_done = True
            n = msg.sfn % 2
            self._fill_tracking(entry['pss']['pos'], msg.sfn,
                                count=self.TRACKING_LOOKAHEAD - 1 + n)

    def _on_mib_failed(self, msg):
        frame_tag = msg.tag.get('frame_tag')
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return

        self.log.append(f'TRACKING: mib_fail {frame_tag}')
        entry['mib']['status'] = 'done'
        entry['sib1']['status'] = 'done'
        self._try_advance_cursor()
        self._try_cleanup()
        self._check_done()

    # ======== SIB1 ========

    def _try_send_sib1(self, frame_tag):
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return
        if entry['mib']['status'] != 'done':
            return
        if entry['sib1']['status'] != 'not_sent':
            return

        p = self.params
        slot1_start = entry['pss']['pos'] + p.N_FFT
        frame_start = slot1_start - p.N_slot
        subf5_start = frame_start + 5 * p.N_subframe
        sib1_offset = subf5_start - self.consume_cursor

        self.system.send_message('buffer_manager', BufferRead(
            offset=sib1_offset,
            length=p.N_subframe,
            dest='sib1_decode',
            tag={'frame_tag': frame_tag,
                 'N_id': self.N_id,
                 'f_d': entry['pss']['f_d'],
                 'n_ant': self.n_ant,
                 'ns': 10,
                 'phich_res': self.phich_res,
                 'sfn': entry['mib']['sfn']}))

        entry['sib1']['status'] = 'pending'
        self.log.append(f'TRACKING: sib1_sent {frame_tag}, SFN={entry["mib"]["sfn"]}')

        self._try_advance_cursor()

    def _on_sib1_done(self, msg):
        frame_tag = msg.tag.get('frame_tag')
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return

        entry['sib1']['status'] = 'done'
        self.sib1_results.append({
            'frame_tag': frame_tag,
            'sfn': entry['mib']['sfn'],
            'sib1_bytes': msg.sib1_bytes
        })
        self.log.append(f'TRACKING: sib1_done {frame_tag}, SFN={entry["mib"]["sfn"]}')

        self._try_cleanup()
        self._check_done()

    def _on_sib1_failed(self, msg):
        frame_tag = msg.tag.get('frame_tag')
        entry = self.tracking_table.get(frame_tag)
        if entry is None:
            return

        self.log.append(f'TRACKING: sib1_fail {frame_tag}')
        entry['sib1']['status'] = 'done'
        self._try_advance_cursor()
        self._try_cleanup()
        self._check_done()

    # ======== Done check ========

    def _check_done(self):
        """Set done if target reached or all work complete after data exhausted."""
        if self._done_event is None:
            return
        # target reached
        if self._sib1_target and len(self.sib1_results) >= self._sib1_target:
            self._done_event.set()
        # data exhausted and no pending work
        elif self._data_exhausted and not self._has_pending_work():
            self._done_event.set()

    def _has_pending_work(self):
        """Any frame still waiting for results?"""
        for entry in self.tracking_table.values():
            if entry['sib1']['status'] != 'done':
                return True
        return False

    # ======== Consume + Cleanup ========

    def _try_advance_cursor(self):
        """Consume past consecutive frames whose SIB1 is at least sent."""
        p = self.params
        sorted_tags = sorted(
            self.tracking_table.keys(),
            key=lambda t: self.tracking_table[t]['pss']['pos'])

        last_sent = None
        for tag in sorted_tags:
            entry = self.tracking_table[tag]
            if entry['sib1']['status'] != 'not_sent':
                last_sent = tag
            else:
                break

        if last_sent is None:
            return

        entry = self.tracking_table[last_sent]
        sfn = entry['mib']['sfn']
        slot1_start = entry['pss']['pos'] + p.N_FFT
        frame_start = slot1_start - p.N_slot

        if sfn is not None and sfn % 2 != 0:
            next_start = frame_start + p.N_frame
        else:
            next_start = frame_start + 2 * p.N_frame

        consume_amount = next_start - self.consume_cursor
        if consume_amount > 0:
            self._do_consume(consume_amount, f'after_{last_sent}')

    def _try_cleanup(self):
        """Remove consecutive done frames from front. Replenish tracking window."""
        sorted_tags = sorted(
            self.tracking_table.keys(),
            key=lambda t: self.tracking_table[t]['pss']['pos'])

        to_delete = []
        for tag in sorted_tags:
            entry = self.tracking_table[tag]
            if entry['sib1']['status'] == 'done':
                to_delete.append(tag)
            else:
                break

        if not to_delete:
            return

        self.log.append(
            f'TRACKING: cleanup {len(to_delete)} frames: '
            + ', '.join(f'{t}(SFN={self.tracking_table[t]["mib"]["sfn"]})'
                        for t in to_delete))

        for tag in to_delete:
            del self.tracking_table[tag]

        # replenish: maintain lookahead count
        while len(self.tracking_table) < self.TRACKING_LOOKAHEAD:
            latest_tag = f'frame_{self._next_frame_idx}'
            if latest_tag not in self.tracking_table:
                break
            max_pos = self.tracking_table[latest_tag]['pss']['pos']
            next_pss = max_pos + 2 * self.params.N_frame
            self._dispatch_pss_tracking(next_pss)

    # ======== Dispatch helpers ========

    def _dispatch_cell_search(self):
        """Send initial search BufferReads covering first half-frame."""
        p = self.params
        pos = 0
        chunk_id = 0
        while pos + p.N_subframe <= p.N_half_frame:
            self.system.send_message('buffer_manager', BufferRead(
                offset=pos,
                length=p.N_subframe,
                dest='cell_search',
                tag={'mode': 'initial_search',
                     'chunk_tag': chunk_id,
                     'pos': self.consume_cursor + pos}))
            pos += p.stride
            chunk_id += 1
        self.log.append(f'CELL_SEARCH: dispatched {chunk_id} chunks')

    def _dispatch_pss_tracking(self, expected_pss_pos):
        """Send one tracking BufferRead for expected PSS position."""
        p = self.params
        self._next_frame_idx += 1
        frame_tag = f'frame_{self._next_frame_idx}'

        entry = self._new_entry(frame_tag)
        entry['pss']['pos'] = expected_pss_pos
        entry['pss']['status'] = 'pending'
        self.tracking_table[frame_tag] = entry

        track_start = expected_pss_pos - 2 * p.N_ofdm_sym
        track_offset = track_start - self.consume_cursor

        self.system.send_message('buffer_manager', BufferRead(
            offset=track_offset,
            length=p.pss_tracking_len,
            dest='cell_search',
            tag={'mode': 'tracking',
                 'frame_tag': frame_tag}))

        self.log.append(f'TRACKING: pss_sent {frame_tag}')

    def _fill_tracking(self, pss_pos, sfn, count):
        """Fill lookahead with tracking entries for future even frames."""
        p = self.params
        first_even_pss = pss_pos + (2 - sfn % 2) * p.N_frame
        for i in range(count):
            ex_pss_pos = first_even_pss + i * 2 * p.N_frame
            self._dispatch_pss_tracking(ex_pss_pos)

    # ======== Helpers ========

    def _new_entry(self, frame_tag):
        return {
            'frame_tag': frame_tag,
            'pss':  {'status': 'not_sent', 'pos': None, 'f_d': None},
            'mib':  {'status': 'not_sent', 'sfn': None},
            'sib1': {'status': 'not_sent', 'sib1_bytes': None}
        }

    def _do_consume(self, amount, reason=''):
        self.log.append(
            f'CONSUME: {reason}, amount={amount}, '
            f'cursor: {self.consume_cursor} → {self.consume_cursor + amount}')
        self.system.send_message('buffer_manager', BufferConsume(length=amount))
        self.consume_cursor += amount
