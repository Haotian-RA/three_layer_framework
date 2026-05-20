import numpy as np
from mib_decode import equalize_1ant, equalize_2ant, equalize_4ant
import numba


def pdsch_crs_symbols(n_ant):
    return {0, 4, 7, 11} if n_ant <= 2 else {0, 1, 4, 7, 8, 11}

def pdsch_extract_indices(cfi, rb_start, L_crbs, crs_syms, nu_shift):
    k_start = rb_start * 12
    N_sc = L_crbs * 12
    k = np.arange(N_sc)
    mod6 = (k_start + k) % 6
    mask = (mod6 != nu_shift) & (mod6 != (nu_shift + 3) % 6)
    sym_list, k_list = [], []

    for l in range(cfi, 14):
        ks = np.arange(k_start, k_start + N_sc)[mask] if l in crs_syms else np.arange(k_start, k_start + N_sc)
        sym_list.append(np.full(len(ks), l, dtype=int)) 
        k_list.append(ks)

    return np.concatenate(sym_list), np.concatenate(k_list)

    

@numba.jit(nopython=True, nogil=True)
def _log_sigmoid(x):
    if x >= 0.0:
        return -np.log1p(np.exp(-x))
    else:
        return x - np.log1p(np.exp(x))


@numba.jit(nopython=True, nogil=True)
def _maxstar_nb(a, b):
    if a <= -1e9:
        return b
    if b <= -1e9:
        return a
    mx = a if a > b else b
    return mx + np.log1p(np.exp(-abs(a - b)))


@numba.jit(nopython=True, nogil=True)
def _bcjr_numba(Ru, R, Ap,
                nt_edges, nt_sign_ap, nt_sign_c0, nt_sign_c1,
                term_edges, term_sign_c0, term_sign_c1,
                bit1_edges, bit0_edges):
    """Full BCJR pass. GIL released.

    nt_edges:  (N_nt, 2) int — (r, s) pairs for non-termination edges
    nt_sign_*: (N_nt,) float — sign multipliers per edge
    term_edges: (N_term, 2) int
    bit1_edges, bit0_edges: (N_b1, 2), (N_b0, 2) — edges where info bit is 1 or 0
    """
    K = Ap.shape[0]
    T = Ru.shape[0]
    Ns = 8
    NEG_INF = -1e10
    N_nt = nt_edges.shape[0]
    N_term = term_edges.shape[0]

    # Gamma
    gamma = np.full((T, Ns, Ns), NEG_INF)

    for t in range(K):
        for e in range(N_nt):
            r = nt_edges[e, 0]; s = nt_edges[e, 1]
            gamma[t, r, s] = (_log_sigmoid(Ap[t] * nt_sign_ap[e])
                            + _log_sigmoid(Ru[t] * nt_sign_c0[e])
                            + _log_sigmoid(R[t] * nt_sign_c1[e]))

    for t in range(K, T):
        for e in range(N_term):
            r = term_edges[e, 0]; s = term_edges[e, 1]
            gamma[t, r, s] = (_log_sigmoid(Ru[t] * term_sign_c0[e])
                            + _log_sigmoid(R[t] * term_sign_c1[e]))

    # Forward
    alpha = np.full((T + 1, Ns), NEG_INF)
    alpha[0, 0] = 0.0
    for t in range(T):
        for s in range(Ns):
            val = NEG_INF
            for r in range(Ns):
                val = _maxstar_nb(val, alpha[t, r] + gamma[t, r, s])
            alpha[t + 1, s] = val

    # Backward
    beta = np.full((T + 1, Ns), NEG_INF)
    beta[T, 0] = 0.0
    for t in range(T - 1, -1, -1):
        for r in range(Ns):
            val = NEG_INF
            for s in range(Ns):
                val = _maxstar_nb(val, beta[t + 1, s] + gamma[t, r, s])
            beta[t, r] = val

    # LLR
    N_b1 = bit1_edges.shape[0]
    N_b0 = bit0_edges.shape[0]
    Lp = np.full(K, NEG_INF)
    Lm = np.full(K, NEG_INF)

    for t in range(K):
        for e in range(N_b1):
            r = bit1_edges[e, 0]; s = bit1_edges[e, 1]
            m = alpha[t, r] + gamma[t, r, s] + beta[t + 1, s]
            Lp[t] = _maxstar_nb(Lp[t], m)
        for e in range(N_b0):
            r = bit0_edges[e, 0]; s = bit0_edges[e, 1]
            m = alpha[t, r] + gamma[t, r, s] + beta[t + 1, s]
            Lm[t] = _maxstar_nb(Lm[t], m)

    result = np.empty(K)
    for i in range(K):
        result[i] = Lm[i] - Lp[i]
    return result

def _build_trellis_edges():
    """Build edge-list trellis for Numba (compact, no 8×8 matrices)."""
    Ns = 8; nu = 3
    nt_list = []; nt_sap = []; nt_sc0 = []; nt_sc1 = []
    bit1_list = []; bit0_list = []
    term_list = []; term_sc0 = []; term_sc1 = []

    for r in range(Ns):
        for s in range(Ns):
            if (r >> 1) == (s & (2**(nu-1)-1)):
                feedback = (r ^ (r >> 1)) & 1
                newbit = (s >> (nu-1)) ^ feedback
                c0 = newbit; c1 = (r ^ (r >> 2) ^ (s >> (nu-1))) & 1
                nt_list.append([r, s])
                nt_sap.append(1.0 - 2.0 * newbit)
                nt_sc0.append(1.0 - 2.0 * c0)
                nt_sc1.append(1.0 - 2.0 * c1)
                if newbit == 1: bit1_list.append([r, s])
                else: bit0_list.append([r, s])
            if r >> 1 == s:
                feedback = (r ^ (r >> 1)) & 1
                c0 = feedback; c1 = (r ^ (r >> 2)) & 1
                term_list.append([r, s])
                term_sc0.append(1.0 - 2.0 * c0)
                term_sc1.append(1.0 - 2.0 * c1)

    return (np.array(nt_list, dtype=np.int64), np.array(nt_sap), np.array(nt_sc0), np.array(nt_sc1),
            np.array(term_list, dtype=np.int64), np.array(term_sc0), np.array(term_sc1),
            np.array(bit1_list, dtype=np.int64), np.array(bit0_list, dtype=np.int64))


@numba.jit(nopython=True, nogil=True)
def _c_sequence_nb(M, c_init):
    """Gold sequence generator. Replaces Python c_sequence."""
    Nc = 1600
    x_1 = np.zeros(M + Nc, dtype=np.uint8)
    x_1[0] = 1
    for n in range(31, M + Nc):
        x_1[n] = x_1[n - 28] ^ x_1[n - 31]

    x_2 = np.zeros(M + Nc, dtype=np.uint8)
    for n in range(31):
        x_2[n] = np.uint8((c_init >> n) & 1)
    for n in range(31, M + Nc):
        x_2[n] = x_2[n - 28] ^ x_2[n - 29] ^ x_2[n - 30] ^ x_2[n - 31]

    result = np.empty(M, dtype=np.uint8)
    for i in range(M):
        result[i] = x_1[Nc + i] ^ x_2[Nc + i]
    return result


@numba.jit(nopython=True, nogil=True)
def _turbo_perm_nb(K, f1, f2):
    """Turbo code interleaver permutation."""
    perm = np.empty(K, dtype=np.int64)
    for i in range(K):
        perm[i] = (f1 * i + f2 * i * i) % K
    return perm


@numba.jit(nopython=True, nogil=True)
def _invert_perm_nb(perm):
    """Invert a permutation array. O(n)."""
    n = len(perm)
    inv = np.empty(n, dtype=np.int64)
    for i in range(n):
        inv[perm[i]] = i
    return inv


@numba.jit(nopython=True, nogil=True)
def _rate_matching_turbo_nb(seq_len, col_perm, rv):
    """Turbo code rate matching interleaver. Replaces Python rate_matching_turbo."""
    C = 32
    D = seq_len
    DUMMY = D + 10000

    R = (D + C - 1) // C
    K_pi = R * C
    N_dummy = K_pi - D

    # build y: dummy symbols + data indices
    y = np.empty(K_pi, dtype=np.int64)
    for i in range(N_dummy):
        y[i] = DUMMY
    for i in range(D):
        y[N_dummy + i] = i

    # column permutation (ports 1 & 2)
    v = np.empty(K_pi, dtype=np.int64)
    for col in range(C):
        src_col = col_perm[col]
        for row in range(R):
            v[col * R + row] = y[row * C + src_col]

    # port 3: QPP interleaver
    v3 = np.empty(K_pi, dtype=np.int64)
    for k in range(K_pi):
        pi_k = (col_perm[k // R] + C * (k % R) + 1) % K_pi
        v3[k] = y[pi_k]

    # offset streams
    v1 = v.copy()
    v2 = v.copy()
    for i in range(K_pi):
        if v2[i] < 10000:
            v2[i] += D
        if v3[i] < 10000:
            v3[i] += 2 * D

    # combine circular buffer
    N_cb = 3 * K_pi
    v_comb = np.zeros(N_cb, dtype=np.int64)
    for i in range(K_pi):
        v_comb[i] = v1[i]
    for i in range(K_pi):
        v_comb[K_pi + 2 * i] = v2[i]
        v_comb[K_pi + 2 * i + 1] = v3[i]

    # RV-dependent start and selection
    k0 = R * (2 * int(np.ceil(N_cb / (8.0 * R))) * rv + 2)

    e = np.empty(3 * D, dtype=np.int64)
    k = 0
    j = 0
    while k < 3 * D:
        idx = (k0 + j) % N_cb
        if v_comb[idx] != DUMMY:
            e[k] = v_comb[idx]
            k += 1
        j += 1

    return e


@numba.jit(nopython=True, nogil=True)
def _crc24a_nb(data, crc_table):
    """CRC-24A check on packed bytes."""
    crc = np.uint32(0)
    for i in range(len(data)):
        idx = np.uint32(((crc >> 16) ^ np.uint32(data[i])) & np.uint32(0xFF))
        crc = ((crc << 8) ^ crc_table[idx]) & np.uint32(0xFFFFFF)
    return crc


@numba.jit(nopython=True, nogil=True)
def _packbits_nb(bits, n_bytes):
    """Pack boolean/uint8 bit array to bytes."""
    result = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(n_bytes):
        val = np.uint8(0)
        for j in range(8):
            idx = i * 8 + j
            if idx < len(bits):
                val = (val << np.uint8(1)) | np.uint8(bits[idx])
            else:
                val = val << np.uint8(1)
        result[i] = val
    return result


# ---- Main DLSCH function: everything in one nogil call ----

@numba.jit(nopython=True, nogil=True)
def _dlsch_full_nb(pdsch_soft, c_init_descramble,
                   tbs, rv, f1, f2,
                   noise_sigma_inv2, llr_scale, num_iterations,
                   col_perm_rm,
                   nt_edges, nt_sign_ap, nt_sign_c0, nt_sign_c1,
                   term_edges, term_sign_c0, term_sign_c1,
                   bit1_edges, bit0_edges,
                   maxR, maxA,
                   crc24_table):
    """Full DLSCH decode: descramble → rate dematch → turbo → CRC. GIL released."""

    n_soft = len(pdsch_soft)

    # ---- 1. PDSCH descramble ----
    c = _c_sequence_nb(n_soft, c_init_descramble)
    descrambled = np.empty(n_soft)
    for i in range(n_soft):
        descrambled[i] = pdsch_soft[i] * (1.0 - 2.0 * c[i])

    # ---- 2. Turbo rate dematch ----
    coded_block_size = tbs + 24 + 4
    collect_len = 3 * coded_block_size

    perm_rm = _rate_matching_turbo_nb(coded_block_size, col_perm_rm, rv)

    coded_flat = np.zeros(collect_len)
    for i in range(collect_len):
        if i < n_soft:
            coded_flat[perm_rm[i]] = descrambled[i]

    # reshape to (3, coded_block_size)
    coded = np.empty((3, coded_block_size))
    for s in range(3):
        for j in range(coded_block_size):
            coded[s, j] = coded_flat[s * coded_block_size + j]

    # ---- 3. LLR scaling ----
    LLRs = np.empty((3, coded_block_size))
    for s in range(3):
        for j in range(coded_block_size):
            LLRs[s, j] = coded[s, j] * noise_sigma_inv2 * llr_scale

    # ---- 4. Turbo decode ----
    K = tbs + 24  # without trellis termination

    perm = _turbo_perm_nb(K, f1, f2)
    perm_inv = _invert_perm_nb(perm)

    # trellis termination shuffling (TS 36.212 §5.1.3.2.2)
    Ru1 = np.empty(K + 3)
    R1 = np.empty(K + 3)
    Ru2 = np.empty(K + 3)
    R2 = np.empty(K + 3)

    for i in range(K):
        Ru1[i] = LLRs[0, i]
        R1[i] = LLRs[1, i]
        Ru2[i] = LLRs[0, perm[i]]
        R2[i] = LLRs[2, i]

    Ru1[K] = LLRs[0, K];     Ru1[K+1] = LLRs[2, K];     Ru1[K+2] = LLRs[1, K+1]
    R1[K] = LLRs[1, K];      R1[K+1] = LLRs[0, K+1];    R1[K+2] = LLRs[2, K+1]
    Ru2[K] = LLRs[0, K+2];   Ru2[K+1] = LLRs[2, K+2];   Ru2[K+2] = LLRs[1, K+3]
    R2[K] = LLRs[1, K+2];    R2[K+1] = LLRs[0, K+3];    R2[K+2] = LLRs[2, K+3]

    # clip
    for i in range(K + 3):
        if Ru1[i] > maxR: Ru1[i] = maxR
        elif Ru1[i] < -maxR: Ru1[i] = -maxR
        if R1[i] > maxR: R1[i] = maxR
        elif R1[i] < -maxR: R1[i] = -maxR
        if Ru2[i] > maxR: Ru2[i] = maxR
        elif Ru2[i] < -maxR: Ru2[i] = -maxR
        if R2[i] > maxR: R2[i] = maxR
        elif R2[i] < -maxR: R2[i] = -maxR

    # turbo iterations
    E2 = np.zeros(K)
    L2 = np.zeros(K)

    for iteration in range(num_iterations):
        A1 = np.empty(K)
        for i in range(K):
            v = E2[perm_inv[i]]
            if v > maxA: v = maxA
            elif v < -maxA: v = -maxA
            A1[i] = v

        L1 = _bcjr_numba(Ru1, R1, A1,
                         nt_edges, nt_sign_ap, nt_sign_c0, nt_sign_c1,
                         term_edges, term_sign_c0, term_sign_c1,
                         bit1_edges, bit0_edges)

        E1 = np.empty(K)
        for i in range(K):
            E1[i] = L1[i] - Ru1[i] - A1[i]

        A2 = np.empty(K)
        for i in range(K):
            v = E1[perm[i]]
            if v > maxA: v = maxA
            elif v < -maxA: v = -maxA
            A2[i] = v

        L2 = _bcjr_numba(Ru2, R2, A2,
                         nt_edges, nt_sign_ap, nt_sign_c0, nt_sign_c1,
                         term_edges, term_sign_c0, term_sign_c1,
                         bit1_edges, bit0_edges)

        for i in range(K):
            E2[i] = L2[i] - Ru2[i] - A2[i]

    # de-interleave
    sib1_LLRs = np.empty(K)
    for i in range(K):
        sib1_LLRs[i] = L2[perm_inv[i]]

    # ---- 5. Hard decision ----
    n_bits = tbs + 24  # K without trellis
    hard_bits = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        hard_bits[i] = np.uint8(1) if sib1_LLRs[i] < 0 else np.uint8(0)

    # ---- 6. Pack bits + CRC-24A ----
    n_bytes = (n_bits + 7) // 8
    hard_bytes = _packbits_nb(hard_bits, n_bytes)

    crc_ok = _crc24a_nb(hard_bytes, crc24_table) == np.uint32(0)

    # return payload bytes (exclude 3-byte CRC)
    payload = hard_bytes[:n_bytes - 3]
    return crc_ok, payload



class CRC24A_Table:
    def __init__(self, crc_poly):
        self._t = np.zeros(256, dtype=np.uint32)
        mask = np.uint32(1 << 23)
        for n in np.arange(256, dtype=np.uint32):
            c = n << 16
            for _ in range(8): c = (crc_poly ^ (c << 1)) if (c & mask) else (c << 1)
            self._t[n] = c & 0xFFFFFF

    def __call__(self, data):
        crc = np.uint32(0)
        for byte in data: crc = ((crc << 8) ^ self._t[((crc >> 16) ^ byte) & 0xFF]) & 0xFFFFFF
        return crc & 0xFFFFFF



def f1f2_table(K):
    table = {
        40:(3,10),48:(7,12),56:(19,42),64:(7,16),72:(7,18),80:(11,20),88:(5,22),96:(11,24),
        104:(7,26),112:(41,84),120:(103,90),128:(15,32),136:(9,34),144:(17,108),152:(9,38),
        160:(21,120),168:(101,84),176:(21,44),184:(57,46),192:(23,48),200:(13,50),208:(27,52),
        216:(11,36),224:(27,56),232:(85,58),240:(29,60),248:(33,62),256:(15,32),264:(17,198),
        272:(33,68),280:(103,210),288:(19,36),296:(19,74),304:(37,76),312:(19,78),320:(21,120),
        328:(21,82),336:(115,84),344:(193,86),352:(21,44),360:(133,90),368:(81,46),376:(45,94),
        384:(23,48),392:(243,98),400:(151,40),408:(155,102),416:(25,52),424:(51,106),432:(47,72),
        440:(91,110),448:(29,168),456:(29,114),464:(247,58),472:(29,118),480:(89,180),488:(91,122),
        496:(157,62),504:(55,84),512:(31,64),528:(17,66),544:(35,68),560:(227,420),576:(65,96),
        592:(19,74),608:(37,76),624:(41,234),640:(39,80),656:(185,82),672:(43,252),688:(21,86),
        704:(155,44),720:(79,120),736:(139,92),752:(23,94),768:(217,48),784:(25,98),800:(17,80),
    }
    return table[K]




class PDSCHDecoding:
    def __init__(self):
        self.EQUALIZE = {1: equalize_1ant, 2: equalize_2ant, 4: equalize_4ant}
        self._n_ant = None; self._crs_syms = None; self._nu_shift = None

    def config(self, N_id, n_ant):
        self._n_ant = n_ant; self._crs_syms = pdsch_crs_symbols(n_ant); self._nu_shift = N_id % 6

    def __call__(self, chunk):
        sym_idx, k_idx = pdsch_extract_indices(chunk.cfi, chunk.rb_start, chunk.L_crbs, self._crs_syms, self._nu_shift)
        r = chunk.symbols[sym_idx, k_idx]; H = chunk.H[:, sym_idx, k_idx]
        x = self.EQUALIZE[self._n_ant](r, H)
        chunk.pdsch_soft = np.zeros(len(x) * 2)
        chunk.pdsch_soft[0::2] = -x.real; chunk.pdsch_soft[1::2] = -x.imag
        return chunk





class DLSCHDecoding:

    COL_PERM_RM = np.array([0,16,8,24,4,20,12,28,2,18,10,26,
                            6,22,14,30,1,17,9,25,5,21,13,29,
                            3,19,11,27,7,23,15,31], dtype=np.int64)

    def __init__(self, noise_sigma=0.3, llr_scale=20, num_iterations=5):
        self.noise_sigma_inv2 = -2.0 / noise_sigma**2
        self.llr_scale = float(llr_scale)
        self.num_iterations = num_iterations

        # trellis tables
        tables = _build_trellis_edges()
        (self._nt_edges, self._nt_sap, self._nt_sc0, self._nt_sc1,
         self._term_edges, self._term_sc0, self._term_sc1,
         self._bit1_edges, self._bit0_edges) = tables

        # CRC-24A table
        crc_obj = CRC24A_Table(0x864CFB)
        self._crc24_table = crc_obj._t.astype(np.uint32)

        self._c_init = None

        # warmup
        dummy_soft = np.zeros(100)
        f1, f2 = f1f2_table(40)
        _dlsch_full_nb(dummy_soft, 0, 16, 0, f1, f2,
                       self.noise_sigma_inv2, self.llr_scale, 1,
                       self.COL_PERM_RM,
                       self._nt_edges, self._nt_sap, self._nt_sc0, self._nt_sc1,
                       self._term_edges, self._term_sc0, self._term_sc1,
                       self._bit1_edges, self._bit0_edges,
                       512.0, 512.0, self._crc24_table)

    def config(self, ns, N_id):
        """Precompute descrambling c_init."""
        self._c_init = (0xFFFF << 14) + ((ns // 2) << 9) + N_id

    def __call__(self, chunk):
        chunk.sib1_decoded = False
        chunk.sib1_bytes = None

        K = chunk.tbs + 24
        f1, f2 = f1f2_table(K)

        crc_ok, payload = _dlsch_full_nb(
            chunk.pdsch_soft, self._c_init,
            chunk.tbs, chunk.rv, f1, f2,
            self.noise_sigma_inv2, self.llr_scale, self.num_iterations,
            self.COL_PERM_RM,
            self._nt_edges, self._nt_sap, self._nt_sc0, self._nt_sc1,
            self._term_edges, self._term_sc0, self._term_sc1,
            self._bit1_edges, self._bit0_edges,
            512.0, 512.0, self._crc24_table)

        if crc_ok:
            chunk.sib1_decoded = True
            chunk.sib1_bytes = payload

        return chunk
