from actor_system import PipelinedActor
from msg_type import SIB1Decoded, SIB1Fail, BufferRelease


class SIB1DecodeActor(PipelinedActor):
    """SIB1 decode. Pipeline: CRS → PCFICH → PDCCH → PDSCH → DLSCH.
    
    PDSCH and DLSCH only run if PDCCH finds DCI (dci_decoded=True).
    """

    def __init__(self, system, buf, pool, graph, node_pipeline):
        super().__init__('sib1_decode', system, pool, graph, node_pipeline[0])
        self._buf = buf
        self._pipeline = node_pipeline

    def _fill_slot(self, slot_idx, msg):
        slot = self._pool.slots[slot_idx]
        data = self._buf.read_protected(msg.pid)
        slot.data[:len(data)] = data
        slot.tag = msg.tag
        slot.N_id = msg.tag['N_id']
        slot.f_d = msg.tag['f_d']
        slot.ns = msg.tag['ns']
        slot.n_ant = msg.tag['n_ant']
        slot.phich_res = msg.tag['phich_res']
        slot.sfn = msg.tag['sfn']
        self.system.send_message('buffer_manager', BufferRelease(pid=msg.pid))

    def on_result(self, msg):
        slot = self._pool.slots[msg.slot]
        if slot.sib1_decoded:
            self.system.send_message('controller', SIB1Decoded(
                sib1_bytes=slot.sib1_bytes, tag=msg.tag))
        else:
            self.system.send_message('controller', SIB1Fail(tag=msg.tag))

    def on_config(self, msg):
        for node in self._pipeline:
            if node.name in msg.params:
                node.func.config(**msg.params[node.name])
