import numpy as np
import numba


def seq_zadoff_chu(u):
    n = np.arange(63)
    d_u = np.exp(-1j * np.pi * u * n * (n + 1) / 63)
    d_u[31] = 0
    return d_u

def zadoff_chu(u, N_FFT):
    zc = seq_zadoff_chu(u)
    re = np.zeros(N_FFT, complex)
    re[N_FFT // 2 - 31:N_FFT // 2 + 32] = zc
    return np.fft.ifft(np.fft.ifftshift(re))

def tilde_s():
    x = np.zeros(31, dtype=np.uint8)
    x[4] = 1
    for i in range(26):
        x[i + 5] = x[i + 2] ^ x[i]
    return 1 - 2.0 * x

def tilde_c():
    x = np.zeros(31, dtype=np.uint8)
    x[4] = 1
    for i in range(26):
        x[i + 5] = x[i + 3] ^ x[i]
    return 1 - 2.0 * x

def tilde_z():
    x = np.zeros(31, dtype=np.uint8)
    x[4] = 1
    for i in range(26):
        x[i + 5] = x[i + 4] ^ x[i + 2] ^ x[i + 1] ^ x[i]
    return 1 - 2.0 * x

def m_01(N_id_1):
    q_prime = N_id_1 // 30
    q = (N_id_1 + q_prime * (q_prime + 1) / 2) // 30
    m_prime = N_id_1 + q * (q + 1) / 2
    m_0 = int(m_prime % 31)
    m_1 = int((m_0 + m_prime // 31 + 1) % 31)
    return (m_0, m_1)

def m_sequence(N_id_1, N_id_2, F, N_FFT):
    m_0, m_1 = m_01(N_id_1)
    ts, tc, tz = tilde_s(), tilde_c(), tilde_z()
    c_0 = np.roll(tc, -N_id_2)
    c_1 = np.roll(tc, -N_id_2 - 3)
    s_0, s_1 = np.roll(ts, -m_0), np.roll(ts, -m_1)
    z_10, z_11 = np.roll(tz, -(m_0 % 8)), np.roll(tz, -(m_1 % 8))
    d = np.zeros(62)
    if F == 0:
        d[0::2] = s_0 * c_0
        d[1::2] = s_1 * c_1 * z_10
    elif F == 1:
        d[0::2] = s_1 * c_0
        d[1::2] = s_0 * c_1 * z_11
    re = np.zeros(N_FFT)
    re[N_FFT // 2 - 31:N_FFT // 2] = d[:31]
    re[N_FFT // 2 + 1:N_FFT // 2 + 32] = d[31:]
    return np.fft.ifft(np.fft.ifftshift(re))


@numba.jit(nopython=True, nogil=True)
def _normalize_and_find_peak(corr, rx_norms, pss_norms):
    """Normalize correlations and find best peak across all 3 PSS roots.

    corr:      (3, L) complex — raw ifft output, already truncated
    rx_norms:  (L,)   float  — sliding window energy
    pss_norms: (3,)   float  — norm of each PSS ref

    Entire function runs with GIL released.
    """
    n_refs = corr.shape[0]
    L = corr.shape[1]

    best_ratio = 0.0
    best_pos = 0
    best_nid2 = 0

    for n in range(n_refs):
        pss_norm = pss_norms[n]

        peak_val = 0.0
        peak_pos_n = 0
        count = 0

        norm_vals = np.empty(L)
        for i in range(L):
            abs_val = abs(corr[n, i])
            if rx_norms[i] > 0:
                norm_vals[i] = abs_val / (rx_norms[i] * pss_norm)
            else:
                norm_vals[i] = 0.0

            if norm_vals[i] > peak_val:
                peak_val = norm_vals[i]
                peak_pos_n = i

            if norm_vals[i] > 0:
                count += 1

        if count > 0:
            pos_vals = np.empty(count)
            j = 0
            for i in range(L):
                if norm_vals[i] > 0:
                    pos_vals[j] = norm_vals[i]
                    j += 1
            pos_vals.sort()

            if count % 2 == 1:
                median_val = pos_vals[count // 2]
            else:
                median_val = (pos_vals[count // 2 - 1] + pos_vals[count // 2]) / 2.0

            ratio = peak_val / median_val if median_val > 0 else 0.0
        else:
            ratio = 0.0

        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = peak_pos_n
            best_nid2 = n

    return best_nid2, best_pos, best_ratio


@numba.jit(nopython=True, nogil=True)
def _sss_search_numba(sss_rx, rx_norm, sig_matrix, norms):
    """Correlate against all 336 SSS candidates. GIL released.

    sss_rx:     (N_FFT,) complex — received SSS symbol
    rx_norm:    float            — energy norm at SSS position
    sig_matrix: (336, N_FFT) complex — precomputed reference signals
    norms:      (336,) float     — norm of each reference

    Returns (best_index, peak_to_median_ratio).
    """
    n_candidates = sig_matrix.shape[0]
    N = sig_matrix.shape[1]

    corr_vals = np.empty(n_candidates)
    for i in range(n_candidates):
        re = 0.0
        im = 0.0
        for j in range(N):
            s = sig_matrix[i, j]
            r = sss_rx[j]
            re += s.real * r.real + s.imag * r.imag
            im += s.real * r.imag - s.imag * r.real
        corr_vals[i] = np.sqrt(re * re + im * im) / (rx_norm * norms[i])

    # find peak
    best_i = 0
    peak_val = corr_vals[0]
    for i in range(1, n_candidates):
        if corr_vals[i] > peak_val:
            peak_val = corr_vals[i]
            best_i = i

    # median
    sorted_vals = corr_vals.copy()
    sorted_vals.sort()
    n = n_candidates
    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    ratio = peak_val / median_val if median_val > 0 else 0.0
    return best_i, ratio




class PSSDetection:

    def __init__(self, params, peak_ratio=5.0):
        self.N_FFT = params.N_FFT
        self.N_CP = params.N_CP
        self.peak_ratio = peak_ratio

        roots = [25, 29, 34]
        self._pss_refs = [zadoff_chu(roots[n], params.N_FFT) for n in range(3)]
        self.pss_norms = np.array([np.linalg.norm(r) for r in self._pss_refs])

        # cache refs for current data length
        self._cached_L = None
        self.refs_fft_conj = None
        self._update_refs(params.N_subframe)

        # warmup numba
        dummy_corr = np.zeros((3, 10), dtype=complex)
        dummy_norms = np.ones(10)
        _normalize_and_find_peak(dummy_corr, dummy_norms, self.pss_norms)

    def _update_refs(self, L):
        self.refs_fft_conj = np.stack([
            np.conj(np.fft.fft(np.pad(self._pss_refs[n], (0, L - self.N_FFT))))
            for n in range(3)
        ])
        self._cached_L = L

    def __call__(self, chunk):
        L = len(chunk.data)
        if L != self._cached_L:
            self._update_refs(L)

        chunk.rx_norms = self._sliding_window_energy(chunk.data)
        rx_fft = np.fft.fft(chunk.data)
        corr = np.fft.ifft(rx_fft[None, :] * self.refs_fft_conj, axis=1)
        L_corr = len(chunk.rx_norms)
        corr = corr[:, :L_corr]

        best_nid2, best_pos, best_ratio = _normalize_and_find_peak(
            corr, chunk.rx_norms, self.pss_norms)

        if best_ratio > self.peak_ratio and best_pos >= self.N_CP + self.N_FFT:
            chunk.pss_detected = True
            chunk.pss_local_index = best_pos
            chunk.N_id_2 = best_nid2

        return chunk

    def _sliding_window_energy(self, rx):
        power = np.abs(rx) ** 2
        cs = np.concatenate(([0], np.cumsum(power)))
        energy = cs[self.N_FFT:] - cs[:len(rx) - self.N_FFT + 1]
        return np.sqrt(energy)
    


class SSSDetection:
    """Hybrid SSS: precomputed refs + Numba nogil for candidate search.

    The 336-candidate correlation loop is the hot path.
    No FFT needed — just complex dot products, perfect for Numba.
    Frequency offset estimation stays in NumPy (small, runs once).
    """

    def __init__(self, params, peak_ratio=5.0):
        self.N_FFT = params.N_FFT
        self.N_CP = params.N_CP
        self.Fs = params.Fs
        self.peak_ratio = peak_ratio

        roots = [25, 29, 34]
        self.pss_refs = {n: zadoff_chu(roots[n], params.N_FFT) for n in range(3)}

        # pre-compute SSS references
        self.sss_refs = {}
        for nid2 in range(3):
            self.sss_refs[nid2] = {}
            for N_id_1 in range(168):
                for F in range(2):
                    idx = N_id_1 + F * 168
                    sig = m_sequence(N_id_1, nid2, F, params.N_FFT)
                    self.sss_refs[nid2][idx] = {
                        'N_id_1': N_id_1,
                        'F': F,
                        'sig': sig,
                        'norm': np.linalg.norm(sig)}

        # stack into numpy arrays for numba
        self._np_refs = {}
        for nid2 in range(3):
            refs = self.sss_refs[nid2]
            ref_keys = list(refs.keys())
            sig_matrix = np.stack([refs[k]['sig'] for k in ref_keys])
            norm_array = np.array([refs[k]['norm'] for k in ref_keys])
            self._np_refs[nid2] = {
                'keys': ref_keys,
                'sig_matrix': sig_matrix,
                'norms': norm_array
            }

        # warmup numba
        _dummy = np.zeros(params.N_FFT, dtype=complex)
        _sss_search_numba(_dummy, 1.0,
                          self._np_refs[0]['sig_matrix'],
                          self._np_refs[0]['norms'])

    def __call__(self, chunk):
        if not chunk.pss_detected:
            return chunk

        sss_start = chunk.pss_local_index - self.N_CP - self.N_FFT
        sss_rx = chunk.data[sss_start:sss_start + self.N_FFT]
        sss_rx_norm = chunk.rx_norms[sss_start]

        refs = self._np_refs[chunk.N_id_2]
        best_i, ratio = _sss_search_numba(
            sss_rx, sss_rx_norm,
            refs['sig_matrix'], refs['norms'])

        if ratio > self.peak_ratio:
            best_idx = refs['keys'][int(best_i)]
            ref = self.sss_refs[chunk.N_id_2][best_idx]
            chunk.sss_detected = True
            chunk.N_id_1 = ref['N_id_1']
            chunk.F = ref['F']
            chunk.f_d = self._estimate_freq_offset(chunk)

        return chunk

    def _estimate_freq_offset(self, chunk):
        N = self.N_FFT
        pss_rx = chunk.data[chunk.pss_local_index:chunk.pss_local_index + N]
        pss_demod = pss_rx * np.conj(self.pss_refs[chunk.N_id_2])
        pl = np.sum(pss_demod[:N // 2])
        pu = np.sum(pss_demod[N // 2:])
        return np.angle(pu * np.conj(pl)) / (2 * np.pi * N // 2) * self.Fs
    


class PSSChunk:
    """simple data structure to test accuracy and no effect on speed test"""
    def __init__(self, data, tag):
        self.data = data
        self.tag = tag

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