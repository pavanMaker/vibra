import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq,ifft,rfft,rfftfreq,irfft
from scipy.signal import detrend, windows, sosfiltfilt, butter,find_peaks
from scipy.signal import firwin, filtfilt
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms
from scipy.integrate import cumulative_trapezoid
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



class Mcc172Backend:
    def __init__(self, board_num=0, sample_rate=5000,  channel=[0, 1],buffer_size = 8192 ):
        self.board_num = board_num
        self.sample_rate = sample_rate
        print("Sample rate:", self.sample_rate)
       
        self.channel = channel
        self.board = mcc172(board_num)
        self.buffer_size = buffer_size
        print("Buffer size:", self.buffer_size)
    def setup(self):
        for ch in self.channel:
            self.board.iepe_config_write(ch, 1)

        self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
        _, self.actual_rate, _ = self.board.a_in_clock_config_read()
        print("Actual sample rate:", self.actual_rate)
        
    #acquistioning
    def start_acquisition(self):
        channel_mask = 0
        for i in self.channel:
            channel_mask |= (1 << i)
        print("channel_mask:", channel_mask)
        self.board.a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)
    #stop acquistion data
    def stop_scan(self):
        self.board.a_in_scan_stop()
        self.board.a_in_scan_cleanup()
        print("Scan stopped and cleaned up.")

    def read_data(self):

        result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=-1)#scanning

        if result and result.data is not None:
            data = np.asarray(result.data) #storing the data of the samples
            print("Raw data type:", type(data))
            print(" Raw data shape:", data.shape)

            ch0_voltage, ch1_voltage = np.zeros(0), np.zeros(0)

            if len(self.channel) == 2:
                if data.ndim == 2:
                    print("for two colums:",data.shape)
                    ch0_voltage = data[0]
                    print("samples of ch0:",len(ch0_voltage))
                    ch1_voltage = data[1]
                    print("samples of ch1:",len(ch1_voltage))
                elif data.ndim == 1:
                    ch0_voltage = data[0::2]
                    print("samples of ch:",ch0_voltage[:20])
                    ch1_voltage = data[1::2]
                    print("samples of ch1:",ch1_voltage[:20])

            elif len(self.channel) == 1:
                only_ch = self.channel[0]
                if data.ndim == 1:
                    if only_ch == 0:
                        ch0_voltage = data
                    else:
                        ch1_voltage = data

            return ch0_voltage, ch1_voltage

        print("❌ No valid data received.")
        return np.zeros(0), np.zeros(0)


    def analyze(self,result_data,sensitivity,fmax_hz,fmin_hz):
        calibiration_fac = 1.0
        
        g_to_m_s2 = 9.80665

        # ───────────── ACCELERATION ─────────────
        acceleration_g = result_data / (sensitivity * calibiration_fac)
        acceleration_waveform = acceleration_g - np.mean(acceleration_g)

        N = len(acceleration_g)
        block_size = 1 / self.actual_rate
        window = windows.hann(N)
        accel_win = acceleration_waveform * window


        windowing_correction = 2.0 / np.sum(window)
        frequencies = rfftfreq(N, block_size)
        print("frequencies:",type(frequencies))
        #acceleration_magnitudes = rfft(accel_win) * windowing_correction
        acceleration_magnitudes = rfft(accel_win)
        # === Frequency Domain RMS Calculation (Parseval's Theorem) ===
        U = np.mean(window ** 2)
        N = len(accel_win)
        fft_result = rfft(accel_win)
        power_spectrum = np.abs(fft_result) ** 2

        # For rfft, double all bins except DC and Nyquist
        if N % 2 == 0:
            power_spectrum[1:-1] *= 2
        else:
            power_spectrum[1:] *= 2

        # Frequencies array, same as used elsewhere in your code
        freqs = rfftfreq(N, block_size)

        # Band-limited mask based on user-selected fmin_hz and fmax_hz
        band = (freqs >= fmin_hz) & (freqs <= fmax_hz)

        # Band-limited power
        band_power = power_spectrum[band]

        # Band-limited RMS calculation (Parseval's theorem, window-corrected)
        rms_fft_acc_band = np.sqrt(np.sum(band_power) / (N ** 2 * U))
        print(f"acceleration_rms (FFT, {fmin_hz} Hz to {fmax_hz} Hz):", rms_fft_acc_band)


        rms_fft_acc = np.sqrt(np.sum(power_spectrum) / (N ** 2 * U))
        print("acceleration_rms from frequency domain (FFT):", rms_fft_acc)


        fft_mags = windowing_correction * np.abs(acceleration_magnitudes)

        # Bandpass filter the acceleration data
        bandpass_freq = (frequencies >= fmin_hz) & (frequencies <= fmax_hz)
        acceleration_magnitudes_filtered = acceleration_magnitudes * bandpass_freq
        #acceleration_magnitudes_filtered =  fft_mags * bandpass_freq
        fft_mags = np.abs(acceleration_magnitudes_filtered) * windowing_correction

        # --- Frequency-Domain Band-Limited RMS for Velocity (mm/s) ---
        U = np.mean(window ** 2)
        frequencies = rfftfreq(N, block_size)
        omega = 2 * np.pi * frequencies
        accel_fft = rfft(accel_win)
        vel_fft = np.where(omega != 0, accel_fft / (1j * omega), 0.0+0.0j)
        vel_power_spectrum = np.abs(vel_fft) ** 2

        if N % 2 == 0:
            vel_power_spectrum[1:-1] *= 2
        else:
            vel_power_spectrum[1:] *= 2

        band = (frequencies >= fmin_hz) & (frequencies <= fmax_hz)
        band_vel_power = vel_power_spectrum[band]

        # RMS in g�s
        rms_fft_vel_band_gs = np.sqrt(np.sum(band_vel_power) / (N ** 2 * U))

        # Convert g�s to mm/s
        g_to_m_s2 = 9.80665
        rms_fft_vel_band = rms_fft_vel_band_gs * g_to_m_s2 * 1000

        print(f"velocity_rms (FFT, {fmin_hz} Hz to {fmax_hz} Hz):", rms_fft_vel_band)


        # ───────────── VELOCITY ─────────────
        omega = 2 * np.pi * frequencies
        
        vel_fft = np.where(omega != 0 ,acceleration_magnitudes_filtered/ (1j * omega),0.0+0.0j)
        #vel_fft[~bandpass_freq] = 0
        vel_fft1= vel_fft * bandpass_freq
        vel_magnitude =  np.abs(vel_fft) * windowing_correction
        vel_fft_mags = vel_magnitude * g_to_m_s2 * 1000
        

        #vel_fft = np.where(omega != 0, acceleration_magnitudes_filtered / (1j * omega), 0.0)
        # vel_fft1 = np.where(omega != 0, acceleration_magnitudes_filtered/ (1j * omega), 0.0)
        # vel_fft = vel_fft1 * g_to_m_s2  # g·s → m/s
        # velocity_magnitudes = vel_fft * bandpass_freq
        # vel_fft_mags = np.abs(velocity_magnitudes) * 1000  # mm/s
        # vel_fft_mags = windowing_correction * np.abs(vel_fft1) * g_to_m_s2 *1000
        # vel_fft_mags *= bandpass_freq    

        vel_time_waveform = irfft(vel_fft) * g_to_m_s2*1000  # mm/s

        # ───────────── DISPLACEMENT ─────────────
        omega_sq = omega ** 2
        disp_fft1 = np.where(omega_sq != 0, acceleration_magnitudes / (-omega_sq), 0.0)
        disp_fft  = windowing_correction * np.abs(disp_fft1)*g_to_m_s2*1000000

        #disp_fft = disp_fft * g_to_m_s2  # g·s² → m

        accel_peak = np.max(np.abs(acceleration_magnitudes_filtered))
        threshold = 0.05 * accel_peak
        disp_mask = np.abs(acceleration_magnitudes_filtered) > threshold
        disp_fft_masked = disp_fft * disp_mask * bandpass_freq
        disp_mags = np.abs(disp_fft_masked) * 1_000_000  # µm

        disp_time_waveform = irfft(disp_fft1) * 1_000_000  # µm

        # ───────────── METRICS (Using endaq.stats.rms) ─────────────
     
    


        peak_indices , _ = find_peaks(fft_mags,height=0.05* np.max(fft_mags))
        a = fft_mags[peak_indices]
        acceleration_peak_reading = np.sum(a)
        print("acceleration overall peak for fft",acceleration_peak_reading)

        acceleration_peak_rms = a/np.sqrt(2)
        overall_rms_peak = np.sum(acceleration_peak_rms)
        print("aceleration overall rms for fft",overall_rms_peak)
        acceleration_peak_rms_2 = acceleration_peak_reading/np.sqrt(2)
        print("acceleration overall rms for fft 2",acceleration_peak_rms_2)
        
        peak_indices_vel,_ = find_peaks(vel_fft_mags, height= 0.05 * np.max(vel_fft_mags))
        v =  vel_fft_mags[peak_indices_vel]
        vel_peak = np.sum(v)
        print("velocity overall peak for fft",vel_peak)
        vel_peak_rms = v / np.sqrt(2)
        vel_peak_reading = np.sum(vel_peak_rms)
        print("velocity overall rms for fft",vel_peak_reading)
        # peak_threshold = fft_mags > threshold
        # acceleration_peak_reading = np.sum(fft_mags[peak_threshold])

        # vel_peak = np.max(np.abs(vel_fft_mags))
        # peak_threshold1 = vel_fft_mags > (0.05 * vel_peak)
        # vel_peak_reading = np.sum(vel_fft_mags[peak_threshold1])

        disp_peak = np.max(np.abs(disp_mags))
        # peak_threshold2 = disp_mags > (0.05 * disp_peak)
        # disp_peak_reading = np.sum(disp_mags[peak_threshold2])

        # displacement_ptp = np.ptp(np.max(disp_mags))  # unnecessary, but kept as-is
        displacement_peak_to = 2 * disp_peak

        dominant_freq = frequencies[np.argmax(fft_mags)]
        peak_freqs = frequencies[peak_indices]
        
        # Extract top frequencies
        if len(a) >= 4:
            top_idx = np.argsort(a)[-4:][::-1]  # descending order
        else:
            top_idx = np.argsort(a)[::-1]       # less than 4 peaks

        # Get top 4 frequencies as list
        top_frequencies = [peak_freqs[i] for i in top_idx]
        print("\nTop 4 FFT Peaks (Frequency, Amplitude):")
        for i in top_idx:
            print(f"  {peak_freqs[i]:.2f} Hz,  {a[i]:.4f}")

        return {
            "acceleration": acceleration_waveform,
            "velocity": vel_time_waveform,
            "displacement": disp_time_waveform,
            "time": np.linspace(0, N * block_size, N, endpoint=False),
            "fft_mags": fft_mags,
            "fft_complex": accel_fft,
            "fft_mags_vel": vel_fft_mags,
            "fft_mags_disp": disp_mags,
            "frequencies": frequencies,
            "acceleration_rms": overall_rms_peak,
            "velocity_rms": vel_peak_reading,
            #"displacement_rms": displacement_rms,
            "acceleration_peak": acceleration_peak_reading,
            "velocity_peak": vel_peak_reading,
            # "displacement_peak": disp_peak_reading,
            # "displacement_peak_to_peak": displacement_ptp,
            "displacement_ptps": displacement_peak_to,
            "dom_freq": dominant_freq,
            "top_frequencies": top_frequencies  # NEW: Multiple frequencies
        }


    
    def get_latest_waveform(self, fmax_hz, fmin_hz, sensitivities):
        ch0_voltage, ch1_voltage = self.read_data()

        print(f"\n📥 Received sensitivities from GUI → CH0: {sensitivities[0]} V/g, CH1: {sensitivities[1]} V/g")


        print(f"\n🔍 Analyzing Channel 0")
        result_ch0 = self.analyze(ch0_voltage,sensitivities[0],fmax_hz=fmax_hz,fmin_hz=fmin_hz) if ch0_voltage.size > 0 else self._empty_result()

        print(f"\n🔍 Analyzing Channel 1")
        result_ch1 = self.analyze(ch1_voltage,sensitivities[1],fmax_hz=fmax_hz,fmin_hz=fmin_hz) if ch1_voltage.size > 0 else self._empty_result()

        return result_ch0, result_ch1

    def _empty_result(self):
        return {
            "acceleration": np.array([]),
            "velocity": np.array([]),
            "displacement": np.array([]),
            "time": np.array([]),
            "acc_peak": 0.0,
            "acc_rms": 0.0,
            "vel_rms": 0.0,
            "disp_ptps": 0.0,
            "dom_freq": 0.0,
            "displacement_ptps": 0.0,
            "frequencies": np.array([]),
            "fft_freqs": np.array([]),
            "fft_mags": np.array([]),
            "fft_complex": np.array([], dtype=complex),
            "freqs_vel": np.array([]),
            "fft_mags_vel": np.array([]),
            "fft_freqs_disp": np.array([]),
            "fft_mags_disp": np.array([]),
            "rms_fft": 0.0
        }
