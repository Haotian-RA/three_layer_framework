import numpy as np
from mib_decode import equalize_1ant, equalize_2ant, equalize_4ant, subblock_interleaver, ConvCoderNumba, CRC16, hamming_dist
import math
from crs_estimate import c_sequence



def pcfich_re_mask(N_rb, N_id):
    N_SC_RB = 12
    N_sc = N_rb * N_SC_RB
    crs_mod3 = (N_id % 6) % 3
    k_bar = (N_SC_RB // 2) * (N_id % (2 * N_rb))
    k_ind = np.zeros(16, dtype=int)
    m = 0
    for q in range(4):
        k_init = (k_bar + ((q * N_rb) // 2) * (N_SC_RB // 2)) % N_sc
        for i in range(6):
            if i % 3 != crs_mod3:
                k_ind[m] = (k_init + i) % N_sc
                m += 1
    return k_ind

def pdcch_reg_table(N_rb, N_id, cfi, n_ant, phich_res):
    nu_shift = N_id % 6
    crs_positions = {nu_shift, (3 + nu_shift) % 6}
    data_offsets_crs = [i for i in range(6) if i not in crs_positions]
    PHICH_RES_MAP = {'1/6': 1/6, '1/2': 1/2, '1': 1, '2': 2}
    all_REG = set(range(2 * N_rb))
    PCFI_REG = {(N_id % (2*N_rb) + (n*N_rb) // 2) % (2*N_rb) for n in range(4)}
    all_REG.difference_update(PCFI_REG)
    all_REG_vec = np.sort(np.array(list(all_REG)))
    n_0 = len(all_REG_vec)
    Ng = PHICH_RES_MAP[phich_res]
    N_group = int(np.ceil(Ng * N_rb / 8))
    for m in range(N_group):
        for i in range(3):
            n_i = (N_id + m + (i * n_0) // 3) % n_0
            all_REG.discard(all_REG_vec[n_i])
    reg_table = []
    for reg_idx in sorted(all_REG):
        k_start = 6 * reg_idx
        reg_table.append((k_start, 0, [k_start + off for off in data_offsets_crs]))
    for l in range(1, cfi):
        has_crs = (l == 1 and n_ant == 4)
        if has_crs:
            for reg_idx in range(2 * N_rb):
                k_start = 6 * reg_idx
                reg_table.append((k_start, l, [k_start + off for off in data_offsets_crs]))
        else:
            for reg_idx in range(3 * N_rb):
                k_start = 4 * reg_idx
                reg_table.append((k_start, l, [k_start, k_start+1, k_start+2, k_start+3]))
    reg_table.sort(key=lambda x: (x[0], x[1]))
    return reg_table, len(reg_table)


def pdcch_deinterleave(N_reg, N_id):
    pdcch_reg = np.arange(N_reg)
    pdcch_reg_cs = np.roll(pdcch_reg, N_id % N_reg)
    perm_table = subblock_interleaver(N_reg)
    pdcch_num_reg = np.zeros(N_reg, dtype=int)
    pdcch_num_reg[perm_table] = pdcch_reg_cs
    return pdcch_num_reg


def dci_1a_size(N_rb):
    N_RIV = math.ceil(math.log2(N_rb * (N_rb + 1) / 2))
    size_1a = 1 + 1 + N_RIV + 5 + 3 + 1 + 2 + 2
    if N_rb >= 50: size_1a += 1
    N_UL_hop = 2 if N_rb >= 50 else 1
    size_0 = 1 + N_UL_hop + N_RIV + 5 + 1 + 2 + 3 + 1
    return max(size_1a, size_0)


def conv_rate_dematch(e_bits, K):
    perm = subblock_interleaver(K)
    coded = np.zeros((3, K), dtype=np.uint8)
    for n in range(3):
        coded[n, perm] = e_bits[n * K:(n + 1) * K]
    return coded


def decode_dci_1a(dci_bits, N_rb):
    N_RIV = math.ceil(math.log2(N_rb * (N_rb + 1) / 2))
    TBS_TABLE = [
        [16,32,56,88,120,152],[24,56,88,144,176,208],[32,72,144,176,208,256],
        [40,104,176,208,256,328],[56,120,208,256,328,408],[72,144,224,328,424,504],
        [88,176,256,392,504,600],[104,224,328,472,584,712],[120,256,392,536,680,808],
        [136,296,456,616,776,936],
    ]
    def b2i(b):
        v = 0
        for x in b: v = (v << 1) | int(x)
        return v
    p = 2
    riv = b2i(dci_bits[p:p+N_RIV]); p += N_RIV
    i_mcs = b2i(dci_bits[p:p+5]); p += 5
    p += 4
    rv = b2i(dci_bits[p:p+2]); p += 2
    tpc = b2i(dci_bits[p:p+2]); p += 2
    L_crbs = riv // N_rb + 1
    rb_start = riv % N_rb
    if L_crbs > N_rb - rb_start:
        L_crbs = N_rb + 1 - riv // N_rb
        rb_start = N_rb - 1 - riv % N_rb
    n_prb = 3 if (tpc & 1) else 2
    tbs = TBS_TABLE[i_mcs][n_prb - 1]
    return {'rb_start': rb_start, 'L_crbs': L_crbs, 'Q_m': 2, 'rv': rv, 'tbs': tbs}


class PCFICHDecoding:
    def __init__(self):
        self.EQUALIZE = {1: equalize_1ant, 2: equalize_2ant, 4: equalize_4ant}
        self.CFI_CODES = {
            1: np.array([0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1], dtype=np.uint8),
            2: np.array([1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0], dtype=np.uint8),
            3: np.array([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1], dtype=np.uint8),
        }
        self._k_ind = None
        self._scramble_seq = None

    def config(self, N_id, N_rb, ns):
        self._k_ind = pcfich_re_mask(N_rb, N_id)
        c_init = (((ns // 2 + 1) * (2 * N_id + 1)) << 9) + N_id
        self._scramble_seq = c_sequence(32, c_init)

    def __call__(self, chunk):
        r = chunk.symbols[0, self._k_ind]
        H = chunk.H[:, 0, self._k_ind]
        eq = self.EQUALIZE[chunk.n_ant](r, H)
        bits = np.zeros(32, dtype=np.uint8)
        bits[0::2] = (eq.real < 0).astype(np.uint8)
        bits[1::2] = (eq.imag < 0).astype(np.uint8)
        bits = bits ^ self._scramble_seq
        best_cfi, best_dist = None, np.inf
        for cfi, code in self.CFI_CODES.items():
            d = np.sum(bits != code)
            if d < best_dist:
                best_dist, best_cfi = d, cfi
        chunk.cfi = best_cfi
        chunk.cfi_hamming = best_dist
        return chunk
    

class PDCCHDecoding:
    def __init__(self):
        self.EQUALIZE = {1: equalize_1ant, 2: equalize_2ant, 4: equalize_4ant}
        self.fec = ConvCoderNumba([0o133, 0o171, 0o165])
        self.crc = CRC16()
        self._N_id = None; self._N_rb = None; self._n_ant = None
        self._dci_payload = None; self._scramble_seq = None
        self._cfi_table = [None]*4

    def config(self, N_id, N_rb, n_ant, phich_res, ns):
        self._N_id, self._N_rb, self._n_ant = N_id, N_rb, n_ant
        self._dci_payload = dci_1a_size(N_rb)
        max_bits = 0
        for cfi in [1, 2, 3]:
            reg_table, N_reg = pdcch_reg_table(N_rb, N_id, cfi, n_ant, phich_res)
            n_cce = N_reg // 9
            deinterleave = pdcch_deinterleave(N_reg, N_id)
            sym_indices = np.zeros(N_reg * 4, dtype=int)
            k_indices = np.zeros(N_reg * 4, dtype=int)
            for i, reg in enumerate(deinterleave):
                _, sym_l, k_list = reg_table[reg]
                for j, k in enumerate(k_list):
                    sym_indices[i*4+j] = sym_l
                    k_indices[i*4+j] = k
            self._cfi_table[cfi] = (N_reg, n_cce, sym_indices, k_indices)
            if N_reg * 8 > max_bits: max_bits = N_reg * 8
        c_init = ((ns // 2) << 9) + N_id
        self._scramble_seq = c_sequence(max_bits, c_init)

    def __call__(self, chunk):
        N_reg, n_cce, sym_idx, k_idx = self._cfi_table[chunk.cfi]
        r = chunk.symbols[sym_idx, k_idx]
        H = chunk.H[:, sym_idx, k_idx]
        x = self.EQUALIZE[self._n_ant](r, H)
        bits = np.zeros(N_reg * 8, dtype=np.uint8)
        bits[0::2] = (x.real < 0).astype(np.uint8)
        bits[1::2] = (x.imag < 0).astype(np.uint8)
        bits = bits ^ self._scramble_seq[:N_reg * 8]
        result = self._blind_search(bits, n_cce)
        if result:
            dci_bits, cost = result
            dci = decode_dci_1a(dci_bits, self._N_rb)
            chunk.dci_decoded = True; chunk.dci_cost = cost
            chunk.rb_start = dci['rb_start']; chunk.L_crbs = dci['L_crbs']
            chunk.Q_m = dci['Q_m']; chunk.rv = dci['rv']; chunk.tbs = dci['tbs']
        return chunk

    def _blind_search(self, pdcch_bits, n_cce):
        BITS_PER_CCE = 72; K = self._dci_payload + 16; tried = set()
        for L in [8, 4]:
            if n_cce < L: continue
            for m in range(2 if L == 8 else 4):
                start = L * (m % (n_cce // L))
                if start + L > n_cce or (L, start) in tried: continue
                tried.add((L, start))
                cce_bits = pdcch_bits[start*BITS_PER_CCE:(start+L)*BITS_PER_CCE]
                coded = conv_rate_dematch(cce_bits, K)
                decoded, cost = self.fec.decode(coded, hamming_dist)
                info = decoded[:self._dci_payload]
                parity = decoded[self._dci_payload:] ^ 1
                if self.crc(np.packbits(np.concatenate([info, parity]))) == 0:
                    return (info, cost)
        return None
    


