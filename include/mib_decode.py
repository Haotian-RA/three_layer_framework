import numpy as np
import numba
from crs_estimate import c_sequence

# ---- Equalization (TS 36.211 §6.3.4.3) ----

def equalize_1ant(r, H):
    return r / H[0]

def equalize_2ant(r, H):
    x = np.zeros_like(r)
    H0, H1 = H[0], H[1]
    scale = np.abs(H0[0::2])**2 + np.abs(H1[0::2])**2
    x[0::2] = (H0[0::2].conj() * r[0::2] + H1[0::2] * r[1::2].conj()) / scale
    x[1::2] = ((-H1[0::2].conj() * r[0::2] + H0[0::2] * r[1::2].conj()) / scale).conj()
    return x

def equalize_4ant(r, H):
    x = np.zeros_like(r)
    H0, H1, H2, H3 = H[0], H[1], H[2], H[3]
    scale02 = np.abs(H0[0::4])**2 + np.abs(H2[0::4])**2
    x[0::4] = (H0[0::4].conj() * r[0::4] + H2[0::4] * r[1::4].conj()) / scale02
    x[1::4] = ((-H2[0::4].conj() * r[0::4] + H0[0::4] * r[1::4].conj()) / scale02).conj()
    scale13 = np.abs(H1[2::4])**2 + np.abs(H3[2::4])**2
    x[2::4] = (H1[2::4].conj() * r[2::4] + H3[2::4] * r[3::4].conj()) / scale13
    x[3::4] = ((-H3[2::4].conj() * r[2::4] + H1[2::4] * r[3::4].conj()) / scale13).conj()
    return x

def descramble(pbch_bits, scramble_seq):
    n_bits = len(pbch_bits)
    best_sec, best_dist, best_bits = None, np.inf, None
    for s in range(4):
        candidate = pbch_bits ^ scramble_seq[s * n_bits:(s + 1) * n_bits]
        reps = candidate.reshape(4, 120)
        dist = 0
        for i in range(4):
            for j in range(i + 1, 4):
                dist += np.sum(reps[i] != reps[j])
        if dist < best_dist:
            best_dist, best_sec, best_bits = dist, s, candidate
    reps = best_bits.reshape(4, 120)
    pbch_bits = (np.sum(reps, axis=0) >= 2).astype(np.uint8)
    return best_sec, pbch_bits


col_perm_table = np.array([1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
                            0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30])

def subblock_interleaver(seq_len, col_perm_table=col_perm_table):
    N_cc = 32
    D = seq_len
    DUMMY = D + 10
    R = (D + N_cc - 1) // N_cc
    N_dummy = R * N_cc - D
    y = np.concatenate((DUMMY * np.ones(N_dummy, dtype=int), np.arange(seq_len)))
    M = np.reshape(y, (R, N_cc))
    P = np.zeros_like(M)
    for n in range(N_cc):
        P[:, n] = M[:, col_perm_table[n]]
    v = np.reshape(P.T, -1)
    return v[v != DUMMY]


def _count_ones(n):
    b = 0
    while n:
        b += n & 1
        n >>= 1
    return b

def _count_bits(n):
    b = 0
    while n:
        b += 1
        n >>= 1
    return b

def hamming_dist(obs, ref):
    return np.sum(ref != obs)


class CRC16:
    def __init__(self, poly=0x1021):
        self._t = np.zeros(256, dtype=np.uint16)
        mask = np.uint16(1 << 15)
        for n in np.arange(256, dtype=np.uint16):
            c = n << 8
            for _ in range(8):
                c = (poly ^ (c << 1)) if (c & mask) else (c << 1)
            self._t[n] = c

    def __call__(self, data):
        crc = np.uint16(0)
        for d in data:
            crc = self._t[((crc >> 8) ^ d) & 0xFF] ^ ((crc << 8) & 0xFFFF)
        return crc & 0xFFFF


# ---- Numba Viterbi ----

@numba.jit(nopython=True, nogil=True)
def _viterbi_decode_numba(d, transition_table, order, Ns):
    """Viterbi decode with forward ACS + backward traceback. GIL released.

    d:                (n_codes, N) uint8 — observed coded bits
    transition_table: (Nt, n_codes) uint8 — encoder output for each transition
    order:            int — constraint length - 1
    Ns:               int — number of states (2^order)
    """
    n_codes = d.shape[0]
    N = d.shape[1]

    # Forward: ACS
    costs = np.zeros(Ns, dtype=np.float64)
    decisions = np.zeros((N, Ns), dtype=np.uint8)

    for n in range(N):
        new_costs = np.full(Ns, 1e30)

        for te in range(Ns):
            for b in range(2):
                t = (te << 1) + b
                ts = t & (Ns - 1)

                # inline hamming distance
                dist = 0
                for m in range(n_codes):
                    if d[m, n] != transition_table[t, m]:
                        dist += 1

                c = costs[ts] + dist
                if c < new_costs[te]:
                    new_costs[te] = c
                    decisions[n, te] = b

        for te in range(Ns):
            costs[te] = new_costs[te]

    # Find best final state
    best_state = 0
    best_cost = costs[0]
    for s in range(1, Ns):
        if costs[s] < best_cost:
            best_cost = costs[s]
            best_state = s

    # Backward: traceback
    decoded = np.zeros(N, dtype=np.uint8)
    state = best_state
    for n in range(N - 1, -1, -1):
        b = decisions[n, state]
        t = (state << 1) + b
        decoded[n] = (t & Ns) >> order
        state = t & (Ns - 1)

    return decoded, best_cost


class ConvCoderNumba:

    def __init__(self, generators):
        self.n_codes = len(generators)
        self.order = max(_count_bits(g) for g in generators) - 1
        self.Ns = 2 << (self.order - 1)
        self.Nt = 2 << self.order
        self._t = np.empty((self.Nt, self.n_codes), dtype=np.uint8)
        for n in range(self.Nt):
            for m in range(self.n_codes):
                self._t[n, m] = _count_ones(n & generators[m]) & 0x1

        # warmup numba
        dummy = np.zeros((self.n_codes, 10), dtype=np.uint8)
        _viterbi_decode_numba(dummy, self._t, self.order, self.Ns)

    def decode(self, d, cost_fun=None):
        return _viterbi_decode_numba(d, self._t, self.order, self.Ns)
    



class PBCHDecoding:

    def __init__(self):
        self.EQUALIZE = {1: equalize_1ant, 2: equalize_2ant, 4: equalize_4ant}
        self._l_idx = None
        self._k_idx = None

    def config(self, N_id):
        self._l_idx, self._k_idx = self.pbch_data_mask(N_id)

    def __call__(self, chunk):
        r = chunk.symbols[self._l_idx, self._k_idx]
        H = chunk.H[:, self._l_idx, self._k_idx]
        chunk.pbch_eq = self.EQUALIZE[chunk.n_ant](r, H)
        chunk.pbch_bits[0::2] = (chunk.pbch_eq.real < 0).astype(np.uint8)
        chunk.pbch_bits[1::2] = (chunk.pbch_eq.imag < 0).astype(np.uint8)
        return chunk

    def pbch_data_mask(self, N_id, N_rb=6, n_symbols=4, N_rb_sc=12):
        N_sc = N_rb * N_rb_sc
        nu_shift = N_id % 6
        crs_offsets = {nu_shift, (nu_shift + 3) % 6}
        l_idx, k_idx = [], []
        for l in range(n_symbols):
            for kk in range(N_sc):
                if l < 2 and (kk % 6) in crs_offsets:
                    continue
                l_idx.append(l)
                k_idx.append(kk)
        return np.array(l_idx), np.array(k_idx)




class BCHDecoding:

    def __init__(self):
        self.BW_TABLE = {0: 6, 1: 15, 2: 25, 3: 50, 4: 75, 5: 100}
        self.PHICH_RES_TABLE = ['1/6', '1/2', '1', '2']
        self.ANT_MASKS = {1: 0x00, 2: 0xFF, 4: 0x33}
        self.perm_table = subblock_interleaver(40)
        self.fec = ConvCoderNumba([0o133, 0o171, 0o165])
        self.crc = CRC16()
        self._scramble_seq = None

    def config(self, N_id):
        self._scramble_seq = c_sequence(1920, N_id)

    def __call__(self, chunk):
        sec, pbch_bits = descramble(chunk.pbch_bits, self._scramble_seq)

        coded = np.zeros((3, 40), dtype=np.uint8)
        for n in range(3):
            coded[n, self.perm_table] = pbch_bits[n * 40:(n + 1) * 40]

        coded_2x = np.concatenate((coded, coded), axis=1)
        decoded_2x, cost = self.fec.decode(coded_2x, None)
        mib_bits = np.concatenate((decoded_2x[40:60], decoded_2x[20:40]))

        mib_bytes = np.packbits(mib_bits)
        mask = self.ANT_MASKS[chunk.n_ant]
        mib_bytes[3] ^= mask
        mib_bytes[4] ^= mask

        if self.crc(mib_bytes) != 0:
            return chunk

        chunk.cost = cost
        chunk.mib_decoded = True
        chunk.sec = sec
        chunk.dl_bw, chunk.phich_dur, chunk.phich_res, chunk.sfn = \
            self.parse_mib(mib_bytes[:3], sec)
        return chunk

    def parse_mib(self, mib_bytes, sec):
        bits = np.unpackbits(mib_bytes)
        bw_idx = int((bits[0] << 2) | (bits[1] << 1) | bits[2])
        dl_bw = self.BW_TABLE[bw_idx]
        phich_dur = 'extended' if bits[3] else 'normal'
        phich_res = self.PHICH_RES_TABLE[int((bits[4] << 1) | bits[5])]
        sfn_mib = 0
        for i in range(8):
            sfn_mib = (sfn_mib << 1) | bits[6 + i]
        sfn = int(sfn_mib) * 4 + sec
        return dl_bw, phich_dur, phich_res, sfn



class MIBChunk:

    def __init__(self, data, tag, N_id, f_d, N_rb=6, ns=1,
                 n_symbols=4, is_slot_start=True):
        N_sc = N_rb * 12

        # input
        self.data = data
        self.tag = tag
        self.N_id = N_id
        self.N_rb = N_rb
        self.N_sc = N_sc
        self.ns = ns
        self.f_d = f_d
        self.n_symbols = n_symbols
        self.is_slot_start = is_slot_start
        self.n_ant = 4

        # CRS outputs
        self.symbols = np.zeros((n_symbols, N_sc), dtype=complex)
        self.H = np.zeros((4, n_symbols, N_sc), dtype=complex)

        # PBCH outputs
        self.pbch_eq = np.zeros(240, dtype=complex)
        self.pbch_bits = np.zeros(480, dtype=np.uint8)

        # BCH outputs
        self.mib_decoded = False
        self.sec = None
        self.dl_bw = None
        self.phich_dur = None
        self.phich_res = None
        self.sfn = None
        self.cost = None
