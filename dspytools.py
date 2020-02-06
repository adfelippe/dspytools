"""
This library is intended to be a simple swiss-army knife for
working with digital signal processing. It is a set of tools
such as filters and functions to find values one would normally
do manually.
"""

import numpy as np
from scipy.signal import butter, lfilter, freqz
from scipy.signal import correlate

"""
    @brief  This functions calculates the center frequency from an FFT

    This method calculates the center frequency of a given FFT and its
    bin values. If the FFT was calculated using some sort of window, try
    using interpolation=True for a more accurate result. Otherwise,
    an weighing algorith is used.

    @param  fft             array_like
                             Raw FFT array.
    @param  bins            array_like
                             Frequency bins of the FFT.
    @param  interpolation   bool, optional
                             If True, calculates center frequency using
                             interpolation.
    @return freq            float
                             Center frequency found in Hertz.
"""
def find_center_frequency(fft, bins, interpolation=False):

    if (interpolation == True):
        fft_peaks_index = []
        fft_half_len = int(len(fft) / 2)
        fft_buf = fft[0:fft_half_len].copy()
        # Get highest energy peak
        fft_peaks_index.append(np.abs(fft_buf).argmax())
        center_frequency_index = fft_peaks_index[0]
        # Get x2 (center), x1 (x2 - 1) and x3 (x2 + 1) and y's (amplitudes)
        y1 = np.abs(fft_buf[center_frequency_index - 1])
        y2 = np.abs(fft_buf[center_frequency_index])
        y3 = np.abs(fft_buf[center_frequency_index + 1])
        x1 = bins[center_frequency_index - 1]
        x2 = bins[center_frequency_index]
        x3 = bins[center_frequency_index + 1]
        # Calculate interpolation to find real frequency
        num = (x1**2 * y3) - (x1**2 * y2) - (x2**2 * y3) + (x2**2 * y1) - (x3**2 * y1) + (x3**2 * y2)
        den = (x3 * y1) - (x1 * y3) - (x2 * y1) + (x2 * y3) - (x3 * y2) + (x1 * y2)
        # Interpolation formula
        freq = -0.5 * (num / den)

    else:
        fft_peaks_value = []
        fft_peaks_index = []
        fft_half_len = int(len(fft) / 2)
        fft_buf = fft[0:fft_half_len].copy()
        # Get highest energy peak
        fft_peaks_index.append(np.abs(fft_buf).argmax())
        index = fft_peaks_index[0]
        fft_peaks_value.append(np.abs(fft_buf[index]))
        # Get higher neighbour
        if (np.abs(fft_buf[index - 1]) > np.abs(fft_buf[index + 1])):
            fft_peaks_index.append(index - 1)
        else:
            fft_peaks_index.append(index + 1)
        index = fft_peaks_index[1]
        fft_peaks_value.append(np.abs(fft_buf[index]))

        # Estimate the center frequency based on energy levels
        # First off, sum the energy levels
        energy_sum = fft_peaks_value[0] + fft_peaks_value[1]
        # Then find the energy factor of the highest peak
        e_factor = fft_peaks_value[0] / energy_sum
        # Now we must subtract highest and its neighbour
        # corresponding frequency to find a delta
        delta_f = bins[fft_peaks_index[0]] - bins[fft_peaks_index[1]]
        # We can now estimate the frequency using the following formula:
        # f_est = (delta_f * e_factor) + neighbour_energy_level
        freq = (delta_f * e_factor) + bins[fft_peaks_index[1]]

    return freq


"""
    @brief  DC component filter

    This method removes DC offset of a given input signal. It can be used
    either for real-time and static acquisitions. For real-time, pass the optional
    realtime argument as True, so a better filter for this scenario is selected.
    If a real-time filter is used in a static scenario, it can cause distortion to
    signals without DC offset.

    @param  signal          array_like
                             Signal to be filtered.
    @param  realtime        bool, optional
                             If True, uses an improved DC filter for real-time.
    @param  R               float, optional
                             If realtime is True, R can be set. R is the filter
                             aggressiveness. The more closer to 1.0, the more aggressive
                             the filter. 0.9 is the default value.
    @return y               array_like
                             Signal filtered, DC offset removed.
"""
def dc_filter(signal, realtime=False, R=0.9):
    # R max value must be 1.0
    if (R > 1.0):
        R = 1.0

    if (realtime == True):
        print('Real-time with R = {}'.format(R))
        y = [0] * len(signal)
        # Real-time DC filter formula: y(n) = x(n) - x(n - 1) + Ry(n - 1)
        for i in range(1, len(signal)):
            y[i] = signal[i] - signal[i - 1] + (R * y[i - 1])
    else:
        signal_mean = np.mean(signal)
        y = signal - signal_mean

    return y


"""
    @brief  Find phase shift between two signals

    This method estimates the phase shift (lag) between two input signals
    at the same frequency. It uses cross-correlation to find a time shift
    and then calculate phase bewteen Y over X.

    @param  x               array_like
                             First signal to be compared.
    @param  y               array_like
                             Second signal to be compared.
    @param  f               float
                             frequency of the signals.
    @return phase           float
                             Phase shift in degrees.
"""
def find_phase_shift(x, y, f):
    xcorr = correlate(x, y)
    xcorr_max = xcorr.argmax()
    dt = np.arange(1 - len(x), len(x))
    recovered_time_shift = dt[xcorr_max]
    phase = 360 * f * recovered_time_shift / 1000
    return phase


"""
    @brief  Apply a Butterworth filter to an input signal

    This method applies a n-th order Butterworth filter to a given
    input signal. Cut-off and sampling frequencies have to be input
    as well. Order is optional, default is 5, which is a high value.

    @param  data            array_like
                             Signal to be filtered.
    @param  cutoff          float
                             Cut-off frequency in Hertz.
                             Butterworth filters have a -3 dB decay at cut-off.
    @param  fs              int
                             Signal sampling frequency in Hertz.
    @param  order           int, optional
                             Order of the butterworth filter.
    @return y               array_like
                             Result of the filtered signal.
"""
def butterworth_lowpass_filter(signal, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, signal)
    return y
