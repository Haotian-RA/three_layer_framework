import numpy as np


class CellSearchSlot:
    """One working slot. Same field interface as PSSChunk so existing
    PSS/SSS functions can read/write without modification.
    """

    def __init__(self, slot_size):
        self.data = np.empty(slot_size, dtype=complex)
        self.tag = None

        # PSS results
        self.rx_norms = None
        self.pss_detected = False
        self.pss_local_index = None
        self.N_id_2 = None

        # SSS results
        self.sss_detected = False   
        self.N_id_1 = None
        self.F = None
        self.f_d = None

    def reset(self):
        self.tag = None
        self.rx_norms = None
        self.pss_detected = False
        self.pss_local_index = None
        self.N_id_2 = None
        self.sss_detected = False
        self.N_id_1 = None
        self.F = None
        self.f_d = None



class MIBSlot:
    """Same field interface as MIBChunk. Pre-allocates arrays CRS/PBCH write into."""

    def __init__(self, slot_size, n_symbols=4, N_rb=6):
        N_sc = N_rb * 12
        self.data = np.empty(slot_size, dtype=complex)
        self.tag = None
        self.N_id = None
        self.f_d = None
        self.n_ant = None

        # CRS writes into these (pre-allocated)
        self.symbols = np.empty((n_symbols, N_sc), dtype=complex)
        self.H = np.empty((4, n_symbols, N_sc), dtype=complex)

        # PBCH writes into this (pre-allocated)
        self.pbch_eq = np.empty(240, dtype=complex)
        self.pbch_bits = np.empty(480, dtype=np.uint8)

        # BCH results
        self.mib_decoded = False
        self.sec = None
        self.dl_bw = None
        self.sfn = None
        self.phich_dur = None
        self.phich_res = None

    def reset(self):
        self.tag = None
        self.N_id = None
        self.f_d = None 
        self.n_ant = None
        self.mib_decoded = False
        self.sec = None
        self.dl_bw = None
        self.sfn = None
        self.phich_dur = None
        self.phich_res = None



class SIB1Slot:
    """Same field interface as SIB1Chunk."""

    def __init__(self, slot_size, n_symbols=14, N_rb=50):
        N_sc = N_rb * 12
        self.data = np.empty(slot_size, dtype=complex)
        self.tag = None

        # metadata — set by _fill_slot
        self.N_id = None
        self.N_rb = None
        self.ns = None
        self.f_d = None
        self.n_ant = None
        self.phich_res = None
        self.sfn = None

        # CRS — pre-allocated, written into
        self.symbols = np.empty((n_symbols, N_sc), dtype=complex)
        self.H = np.empty((4, n_symbols, N_sc), dtype=complex)

        # PCFICH
        self.cfi = None
        self.cfi_hamming = None

        # PDCCH
        self.dci_decoded = False
        self.dci_cost = None
        self.rb_start = None
        self.L_crbs = None
        self.Q_m = None
        self.rv = None
        self.tbs = None

        # PDSCH
        self.pdsch_soft = None

        # DLSCH
        self.sib1_decoded = False
        self.sib1_bytes = None

    def reset(self):
        self.tag = None
        self.N_id = None
        self.N_rb = None
        self.ns = None
        self.f_d = None
        self.n_ant = None
        self.phich_res = None
        self.sfn = None
        self.cfi = None
        self.cfi_hamming = None
        self.dci_decoded = False
        self.dci_cost = None
        self.rb_start = None
        self.L_crbs = None
        self.Q_m = None
        self.rv = None
        self.tbs = None
        self.pdsch_soft = None
        self.sib1_decoded = False
        self.sib1_bytes = None