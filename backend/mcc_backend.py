import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq,ifft,rfft,rfftfreq,irfft
from scipy.signal import detrend, windows, sosfiltfilt, butter
from scipy.signal import firwin, filtfilt
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms
from scipy.integrate import cumulative_trapezoid
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



class Mcc172Backend:
    def __init__(self, board_num=0, sample_rate=5000,  channel=[0, 1],buffer_size =8192):
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
            print("🔧 Raw data type:", type(data))
            print("📏 Raw data shape:", data.shape)

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

    def analyze(self, data, sensitivity, fmax_hz, fmin_hz):
        if data is None or len(data) < self.buffer_size:
            return self._empty_result()
        
        print(f"raw data length: {len(data)} samples and sensitivity: {sensitivity} V/g")

        fs = self.actual_rate
        print(f"type of data: {type(data)} and Actual Sample Rate: {fs} Hz")
        dt = 1.0 / fs
        N = len(data)
        time = np.linspace(0, (N - 1) * dt, N)

    
        acc_g = data / float(sensitivity)

    
        acc_g = acc_g - np.mean(acc_g)

    
        # numtaps = 513
        # fir_bp = firwin(
        #     numtaps,
        #     [fmin_hz, fmax_hz],
        #     fs=fs,
        #     pass_zero=False,
        #     window="hann"
        # )
        acc_bp = filtfilt(fir_bp, 1.0, acc_g)
        print(f"Applied FIR bandpass filter: {fmin_hz} Hz to {fmax_hz} Hz")
        print(f" mean after filtering: {np.mean(acc_bp):.6f} g")

        window = windows.hann(N)
        acc_win = acc_bp * window
        acc_fft = rfft(acc_win)
        acc_mag = np.abs(acc_fft) * (4.0 / N)
        freqs = rfftfreq(N, dt)

        # DIAGNOSTIC: Print max amplitude with frequency for acceleration
        acc_max_idx = int(np.argmax(acc_mag))
        print(f"🎯 ACCELERATION - Dominant Frequency: {freqs[acc_max_idx]:.2f} Hz, Max Amplitude: {acc_mag[acc_max_idx]:.6f} g")

    
        acc_ms2 = acc_bp * 9.80665

        vel_raw = np.zeros_like(acc_mag)
        for i in range(1, N):
            vel_raw[i] = vel_raw[i-1] + 0.5 * (acc_ms2[i-1] + acc_ms2[i]) * dt

    
        vel_raw = detrend(vel_raw, type="linear")

        # Applying Same fir bp on velocity
        vel_bp = filtfilt(fir_bp, 1.0, vel_raw)

        vel_mm = vel_bp * 1000.0  # m/s → mm/s

        # -------------------------------------------------------
        # 6) VELOCITY FFT
        # -------------------------------------------------------
        vel_win = vel_mm * window
        vel_fft = rfft(vel_win)
        vel_mag = np.abs(vel_fft) * (4.0 / N)
        
        # DIAGNOSTIC: Print max amplitude with frequency for velocity
        vel_max_idx = int(np.argmax(vel_mag))
        print(f"🎯 VELOCITY - Dominant Frequency: {freqs[vel_max_idx]:.2f} Hz, Max Amplitude: {vel_mag[vel_max_idx]:.6f} mm/s")


        # -------------------------------------------------------
        # 7) VEL → DISP (Second Integration)
        # -------------------------------------------------------
        disp_raw = np.zeros_like(vel_mm)
        for i in range(1, N):
            disp_raw[i] = disp_raw[i-1] + 0.5 * (vel_mm[i-1] + vel_mm[i]) * dt

        disp_raw = detrend(disp_raw, type="linear")

        # Apply SAME FIR BP on displacement
        disp_bp = filtfilt(fir_bp, 1.0, disp_raw)

        disp_um = disp_bp * 1000.0  # mm → µm

        # -------------------------------------------------------
        # 8) DISPLACEMENT FFT
        # -------------------------------------------------------
        disp_win = disp_um * window
        disp_fft = rfft(disp_win)
        disp_mag = np.abs(disp_fft) * (4.0 / N)
        
        # DIAGNOSTIC: Print max amplitude with frequency for displacement
        disp_max_idx = int(np.argmax(disp_mag))
        print(f"🎯 DISPLACEMENT - Dominant Frequency: {freqs[disp_max_idx]:.2f} Hz, Max Amplitude: {disp_mag[disp_max_idx]:.6f} µm")

        # -------------------------------------------------------
        # 9) METRICS
        # -------------------------------------------------------
        acc_rms = float(np.sqrt(np.mean(acc_bp**2)))
        acc_peak = float(np.max(np.abs(acc_bp)))

        vel_rms = float(np.sqrt(np.mean(vel_mm**2)))
        vel_peak = float(np.max(np.abs(vel_mm)))

        disp_pp = float(np.ptp(disp_um))
        disp_peak = float(np.max(np.abs(disp_um)))

        dom_idx = int(np.argmax(vel_mag))
        dom_freq = float(freqs[dom_idx])

        # -------------------------------------------------------
        # 10) RETURN RESULT
        # -------------------------------------------------------
        return {
            "time": time,

            "acceleration": acc_bp,
            "velocity": vel_mm,
            "displacement": disp_um,

            "frequencies": freqs,
            "fft_mags": acc_mag,
            "fft_mags_vel": vel_mag,
            "fft_mags_disp": disp_mag,

            "acc_peak": acc_peak,
            "acc_rms": acc_rms,

            "vel_peak": vel_peak,
            "velocity_rms": vel_rms,

            "disp_peak": disp_peak,
            "displacement_ptps": disp_pp,

            "dom_freq": dom_freq,
        }


    
    
    def get_latest_waveform(self,fmax_hz,fmin_hz,sensitivities =None):
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
            "disp_pp": 0.0,
            "dom_freq": 0.0,
            "fft_freqs": np.array([]),
            "fft_mags": np.array([]),
            "freqs_vel": np.array([]),
            "fft_mags_vel": np.array([]),
            "fft_freqs_disp": np.array([]),
            "fft_mags_disp": np.array([]),
            "rms_fft": 0.0
        }
