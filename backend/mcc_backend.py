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
        result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)#scanning

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

    def analyze(self, result_data, sensitivity, fmax_hz, fmin_hz):

        # ------------------------------------------------------------
        # 0) Safety: empty input
        # ------------------------------------------------------------
        if result_data is None or len(result_data) == 0:
            return self._empty_result()

        # ------------------------------------------------------------
        # 1) Prepare data
        # ------------------------------------------------------------
        acc_volts = np.asarray(result_data, dtype=float)
       

        fs = self.actual_rate
        N = len(acc_volts)
        dt = 1.0 / fs

       

        print(f"🔧 Analyzing {N} samples at {fs} Hz with sensitivity {sensitivity} V/g")

        time = np.linspace(0, (N - 1) * dt, N)

        # ------------------------------------------------------------
        # 2) Convert volts → g, remove DC
        # ------------------------------------------------------------
        acc_g = (acc_volts / float(sensitivity))
        acc_g = acc_g - np.mean(acc_g)

        # ------------------------------------------------------------
        # 3) FIR Band-Pass filter for acceleration
        # ------------------------------------------------------------
        numtaps_bp = 513
        fir_bp = firwin(numtaps_bp, [fmin_hz, fmax_hz], fs=fs,
                        pass_zero=False, window='hann')

        acc_bp = filtfilt(fir_bp, 1.0, acc_g)

        print("DBG acc_bp: mean=", np.mean(acc_bp),
            "rms=", np.sqrt(np.mean(acc_bp**2)))

        # ------------------------------------------------------------
        # 4) FFT of acceleration (correct Hann normalization)
        # ------------------------------------------------------------
        window = windows.hann(N)
        U = np.sum(window) / N       # coherent gain
        acc_win = acc_bp * window

        acc_fft = rfft(acc_win)
        freqs = rfftfreq(N, dt)
        M = len(acc_fft)

        acc_spec = np.abs(acc_fft) / (N * U)

        # single-sided spectrum correction
        acc_mag = np.zeros_like(acc_spec)
        acc_mag[0] = acc_spec[0]

        if N % 2 == 0:   # even N → Nyquist exists
            acc_mag[1:-1] = 2.0 * acc_spec[1:-1]
            acc_mag[-1] = acc_spec[-1]
        else:
            acc_mag[1:] = 2.0 * acc_spec[1:]

        # ------------------------------------------------------------
        # 5) Acc → Vel integration (m/s)
        # ------------------------------------------------------------
        acc_ms2 = acc_bp * 9.80665
        acc_ms2 = detrend(acc_ms2, type='linear')

        vel = cumulative_trapezoid(acc_ms2, dx=dt, initial=0.0)
        vel = detrend(vel, type='linear')

        # ------------------------------------------------------------
        # 6) POST-INTEGRATION HPF (drift removal only!)
        #    *** IMPORTANT FIX ***
        # ------------------------------------------------------------
        numtaps_hp = 513
        hp_cut_vel = 2.0   # DO NOT USE fmin_hz HERE!
        fir_hp_vel = firwin(numtaps_hp, hp_cut_vel, fs=fs,
                            pass_zero=False, window='hann')

        vel_hp = filtfilt(fir_hp_vel, 1.0, vel)
        vel_mm = vel_hp * 1000.0   # m/s → mm/s

        print("DBG vel_hp: rms=", np.sqrt(np.mean(vel_mm**2)))

        # ------------------------------------------------------------
        # 7) FFT of velocity
        # ------------------------------------------------------------
        vel_win = vel_mm * window
        vel_fft = rfft(vel_win)
        vel_spec = np.abs(vel_fft) / (N * U)

        vel_mag = np.zeros_like(vel_spec)
        vel_mag[0] = vel_spec[0]

        if N % 2 == 0:
            vel_mag[1:-1] = 2.0 * vel_spec[1:-1]
            vel_mag[-1] = vel_spec[-1]
        else:
            vel_mag[1:] = 2.0 * vel_spec[1:]

        # ------------------------------------------------------------
        # 8) Vel → Disp integration
        # ------------------------------------------------------------
        disp_mm = cumulative_trapezoid(vel_mm, dx=dt, initial=0.0)
        disp_mm = detrend(disp_mm, type='linear')

        disp_hp = filtfilt(fir_hp_vel, 1.0, disp_mm)
        disp_um = disp_hp * 1000.0  # mm → µm

        # ------------------------------------------------------------
        # 9) FFT of displacement
        # ------------------------------------------------------------
        disp_win = disp_um * window
        disp_fft = rfft(disp_win)
        disp_spec = np.abs(disp_fft) / (N * U)

        disp_mag = np.zeros_like(disp_spec)
        disp_mag[0] = disp_spec[0]

        if N % 2 == 0:
            disp_mag[1:-1] = 2.0 * disp_spec[1:-1]
            disp_mag[-1] = disp_spec[-1]
        else:
            disp_mag[1:] = 2.0 * disp_spec[1:]

        # ------------------------------------------------------------
        # 10) Metrics
        # ------------------------------------------------------------
        acc_rms = float(np.sqrt(np.mean(acc_bp**2)))
        vel_rms = float(np.sqrt(np.mean(vel_mm**2)))
        disp_pp = float(np.ptp(disp_um))

        dom_idx = int(np.argmax(vel_mag))
        dom_freq = float(freqs[dom_idx])

        top4_idx = np.argsort(vel_mag)[-4:][::-1]
        top4 = [(float(freqs[i]), float(vel_mag[i])) for i in top4_idx]

        # ------------------------------------------------------------
        # 11) Return dictionary
        # ------------------------------------------------------------
        return {
            "time": time,
            "acceleration": acc_bp,
            "velocity": vel_mm,
            "displacement": disp_um,

            "frequencies": freqs,
            "fft_mags": acc_mag,
            "fft_mags_vel": vel_mag,
            "fft_mags_disp": disp_mag,

            "acc_rms": acc_rms,
            "velocity_rms": vel_rms,
            "displacement_ptps": disp_pp,

            "acc_peak": float(np.max(np.abs(acc_bp))),
            "vel_peak": float(np.max(np.abs(vel_mm))),
            "disp_peak": float(np.max(np.abs(disp_um))),

            "dom_freq": dom_freq,
            "velocity_top4_peaks": top4,
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



    