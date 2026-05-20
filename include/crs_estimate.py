import numpy as np
import numba


# Pseudo-random sequence (TS 36.211 §7.2)
def c_sequence(M, c_init, Nc=1600):
    """Gold sequence of length M, seeded by c_init."""
    x_1 = np.zeros(M + Nc, dtype=np.uint8)
    x_1[0] = 1
    for n in range(31, M + Nc):
        x_1[n] = x_1[n - 28] ^ x_1[n - 31]

    x_2 = np.zeros(M + Nc, dtype=np.uint8)
    for n in range(31):
        x_2[n] = (c_init & (1 << n)) >> n
    for n in range(31, M + Nc):
        x_2[n] = x_2[n - 28] ^ x_2[n - 29] ^ x_2[n - 30] ^ x_2[n - 31]

    return x_1[Nc:] ^ x_2[Nc:]

# CRS sequence and location (TS 36.211 §6.10.1) 
def crs_seq(ns, l, N_id, N_cp=1, N_RB_MAX_DL = 110):
    """Cell-specific reference signal sequence."""
    c_init = (((7 * (ns + 1) + l + 1) * (2 * N_id + 1)) << 10) + 2 * N_id + N_cp
    c = c_sequence(4 * N_RB_MAX_DL, c_init)
    return np.sqrt(0.5) * ((1 - 2.0 * c[0::2]) + 1j * (1 - 2.0 * c[1::2]))


def extract_OFDM(ofdm_symbol, N_rb, N_rb_sc=12):
    """FFT + extract center N_rb*N_rb_sc subcarriers, skipping DC."""
    re = np.fft.fftshift(np.fft.fft(ofdm_symbol))
    N_sc = N_rb * N_rb_sc
    N_FFT = len(ofdm_symbol)
    active_sc = np.concatenate((
        np.arange(N_FFT // 2 - N_sc // 2, N_FFT // 2),
        np.arange(N_FFT // 2 + 1, N_FFT // 2 + N_sc // 2 + 1)
    ))
    return re[active_sc]

def crs_syms_and_k(N_rb, p, l, ns, N_id, N_cp=1, N_symb_DL=7, N_RB_MAX_DL = 110):
    """CRS subcarrier indices and values for antenna port p,
    symbol l, slot ns. Returns (k, crs) or (None, None)."""
    nu = -1
    if p == 0:
        if l == 0:       nu = 0
        elif l == N_symb_DL - 3: nu = 3
    if p == 1:
        if l == 0:       nu = 3
        elif l == N_symb_DL - 3: nu = 0
    if p == 2 and l == 1:
        nu = 3 * (ns % 2)
    if p == 3 and l == 1:
        nu = 3 + 3 * (ns % 2)

    if nu == -1:
        return None, None

    nu_shift = N_id % 6
    m = np.arange(2 * N_rb)
    k = 6 * m + (nu + nu_shift) % 6

    mm = m + N_RB_MAX_DL - N_rb
    rr = crs_seq(ns, l, N_id, N_cp)
    crs_syms = rr[mm]

    return k, crs_syms


@numba.jit(nopython=True, nogil=True)
def _pre_fft_nb(data, f_d, Fs, sym_starts, N_FFT, n_symbols):
    """Freq correct + stack OFDM symbols. One Numba call, zero temp allocation."""
    phase_inc = -2.0 * np.pi * f_d / Fs
    ofdm = np.empty((n_symbols, N_FFT), dtype=np.complex128)
    for l in range(n_symbols):
        start = sym_starts[l]
        for i in range(N_FFT):
            idx = start + i
            phase = phase_inc * idx
            ofdm[l, i] = data[idx] * (np.cos(phase) + 1j * np.sin(phase))
    return ofdm


@numba.jit(nopython=True, nogil=True)
def _post_fft_nb(fft_all, fft_indices, symbols, H, n_ant,
                 n_crs_entries, crs_port, crs_l, crs_k, crs_conj, crs_freq_idx,
                 n_crs_per, N_sc,
                 n_time_entries, time_port, time_l, time_nearest):
    """Extract subcarriers + channel estimation + time interpolation. One Numba call."""
    n_symbols = fft_all.shape[0]

    # ---- Extract active subcarriers ----
    for l in range(n_symbols):
        for i in range(N_sc):
            symbols[l, i] = fft_all[l, fft_indices[i]]

    # ---- Channel estimation at CRS positions ----
    for e in range(n_crs_entries):
        p = crs_port[e]
        if p >= n_ant:
            continue
        l = crs_l[e]

        h_crs = np.empty(n_crs_per, dtype=np.complex128)
        for i in range(n_crs_per):
            h_crs[i] = symbols[l, crs_k[e, i]] * crs_conj[e, i]

        for i in range(N_sc):
            H[p, l, i] = h_crs[crs_freq_idx[e, i]]

    # ---- Time interpolation ----
    for e in range(n_time_entries):
        p = time_port[e]
        if p >= n_ant:
            continue
        l = time_l[e]
        nearest = time_nearest[e]
        for i in range(N_sc):
            H[p, l, i] = H[p, nearest, i]


class CRSChannelEstimation:

    def __init__(self, params):
        self.N_FFT = params.N_FFT
        self.N_CP = params.N_CP
        self.N_CP_extra = params.N_CP_extra
        self.Fs = params.Fs

        self._n_symbols = None
        self._N_sc = None
        self._sym_starts = None
        self._fft_indices = None
        self._n_crs_per = None

        self._n_crs_entries = 0
        self._crs_port = None
        self._crs_l = None
        self._crs_k = None
        self._crs_conj = None
        self._crs_freq_idx = None

        self._n_time_entries = 0
        self._time_port = None
        self._time_l = None
        self._time_nearest = None

    def config(self, N_id, N_rb, ns, n_symbols, is_slot_start=False):
        self._n_symbols = n_symbols
        N_sc = N_rb * 12
        self._N_sc = N_sc

        # symbol start offsets
        self._sym_starts = np.empty(n_symbols, dtype=np.int64)
        start = 0
        for l in range(n_symbols):
            cp = self.N_CP
            if l % 7 == 0 and is_slot_start:
                cp += self.N_CP_extra
            start += cp
            self._sym_starts[l] = start
            start += self.N_FFT

        # FFT extraction indices (raw FFT, no fftshift)
        N_FFT = self.N_FFT
        active_shifted = np.concatenate((
            np.arange(N_FFT // 2 - N_sc // 2, N_FFT // 2),
            np.arange(N_FFT // 2 + 1, N_FFT // 2 + N_sc // 2 + 1)
        ))
        self._fft_indices = ((active_shifted - N_FFT // 2) % N_FFT).astype(np.int64)

        # build CRS table
        crs_table = [[] for _ in range(4)]
        for p in range(4):
            for l in range(n_symbols):
                local_l = l % 7
                local_ns = ns + (l // 7)
                k, crs = crs_syms_and_k(N_rb, p, local_l, local_ns, N_id)
                if k is None:
                    continue
                freq_idx = np.array([np.argmin(np.abs(k - kk)) for kk in range(N_sc)])
                crs_table[p].append((l, k, crs, freq_idx))

        # flatten CRS table
        total_entries = sum(len(e) for e in crs_table)
        n_crs_per = 2 * N_rb
        self._n_crs_per = n_crs_per

        self._crs_port = np.empty(total_entries, dtype=np.int64)
        self._crs_l = np.empty(total_entries, dtype=np.int64)
        self._crs_k = np.empty((total_entries, n_crs_per), dtype=np.int64)
        self._crs_conj = np.empty((total_entries, n_crs_per), dtype=np.complex128)
        self._crs_freq_idx = np.empty((total_entries, N_sc), dtype=np.int64)

        idx = 0
        for p in range(4):
            for l, k, crs, freq_idx in crs_table[p]:
                self._crs_port[idx] = p
                self._crs_l[idx] = l
                self._crs_k[idx, :] = k
                self._crs_conj[idx, :] = np.conj(crs)
                self._crs_freq_idx[idx, :] = freq_idx
                idx += 1
        self._n_crs_entries = idx

        # flatten time interpolation table
        time_entries = []
        for p in range(4):
            crs_syms = [entry[0] for entry in crs_table[p]]
            if not crs_syms:
                continue
            for l in range(n_symbols):
                if l not in crs_syms:
                    nearest = crs_syms[np.argmin([abs(l - s) for s in crs_syms])]
                    time_entries.append((p, l, nearest))

        n_time = len(time_entries)
        self._time_port = np.empty(n_time, dtype=np.int64)
        self._time_l = np.empty(n_time, dtype=np.int64)
        self._time_nearest = np.empty(n_time, dtype=np.int64)
        for i, (p, l, nearest) in enumerate(time_entries):
            self._time_port[i] = p
            self._time_l[i] = l
            self._time_nearest[i] = nearest
        self._n_time_entries = n_time

    def __call__(self, chunk):
        # 1. Freq correct + stack symbols (one Numba call, nogil)
        ofdm_stack = _pre_fft_nb(chunk.data, chunk.f_d, self.Fs,
                                  self._sym_starts, self.N_FFT, self._n_symbols)

        # 2. Batch FFT (one NumPy call, releases GIL)
        fft_all = np.fft.fft(ofdm_stack, axis=1)

        # 3. Extract + estimate + interpolate (one Numba call, nogil)
        _post_fft_nb(fft_all, self._fft_indices, chunk.symbols, chunk.H, chunk.n_ant,
                     self._n_crs_entries,
                     self._crs_port, self._crs_l, self._crs_k,
                     self._crs_conj, self._crs_freq_idx,
                     self._n_crs_per, self._N_sc,
                     self._n_time_entries,
                     self._time_port, self._time_l, self._time_nearest)

        return chunk