import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, windows
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



class Mcc172Backend:
    def __init__(self, board_num=0, sample_rate=51200, sensitivity=0.1, channel=[0, 1]):
        self.board_num = board_num
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        self.channel = channel
        self.board = mcc172(board_num)
        self.buffer_size = None
        self.actual_rate = None

    def setup(self):
        for ch in self.channel:
            self.board.iepe_config_write(ch, 1)

        self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
        _, self.actual_rate, _ = self.board.a_in_clock_config_read()
        self.buffer_size = 196608
        print("buffer size", self.buffer_size)

    def start_acquisition(self):
        channel_mask = 0
        for i in self.channel:
            channel_mask |= (1 << i)
        print("channel_mask:", channel_mask)
        self.board.a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)

    def stop_scan(self):
        self.board.a_in_scan_stop()
        self.board.a_in_scan_cleanup()
        print("Scan stopped and cleaned up.")

    def read_data(self):
        result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)

        if result and result.data is not None:
            data = np.asarray(result.data)
            print("🔧 Raw data type:", type(data))
            print("📏 Raw data shape:", data.shape)

            if data.ndim == 2 and data.shape[0] == 2:
                ch0_voltage = data[0].flatten()
                ch1_voltage = data[1].flatten()
            elif data.ndim == 1 and data.shape[0] == 2 * self.buffer_size:
                ch0_voltage = data[0::2]
                ch1_voltage = data[1::2]
                print("✅ Split interleaved data manually.")
            elif data.ndim == 1 and data.shape[0] > 0:
                print("⚠️ Only one channel returned, assuming CH0 only.")
                ch0_voltage = data.flatten()
                ch1_voltage = np.zeros_like(ch0_voltage)
            else:
                print("❌ Unexpected data format. Filling with zeros.")
                ch0_voltage = np.zeros(self.buffer_size)
                ch1_voltage = np.zeros(self.buffer_size)

            # Safe voltage stats logging
            print("\n📊 CH0 Voltage Stats:")
            if ch0_voltage.size > 0:
                print(f"   → Min: {np.min(ch0_voltage):.3f} V")
                print(f"   → Max: {np.max(ch0_voltage):.3f} V")
                print(f"   → Mean: {np.mean(ch0_voltage):.3f} V")
            else:
                print("   ⚠️ No CH0 data available")

            print("\n📊 CH1 Voltage Stats:")
            if ch1_voltage.size > 0:
                print(f"   → Min: {np.min(ch1_voltage):.3f} V")
                print(f"   → Max: {np.max(ch1_voltage):.3f} V")
                print(f"   → Mean: {np.mean(ch1_voltage):.3f} V")
            else:
                print("   ⚠️ No CH1 data available")

            return ch0_voltage, ch1_voltage

        print("❌ No valid data received.")
        return np.zeros(0), np.zeros(0)

    def analyze(self, result_data,fmax_hz=500,fmin_hz=1):
        # Convert voltage to acceleration in g
        acceleration_g = result_data / self.sensitivity
        acceleration_g = detrend(acceleration_g, type='linear')  # Remove trend
        acceleration_m_s2 = acceleration_g * 9.80665  # Convert to m/s²

        # Time setup
        N = len(acceleration_m_s2)
        T = 1 / self.actual_rate
        time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
        df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])

        # Integrate to velocity and displacement
        integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)
        velocity_m_s = integrated[1]["acceleration"].to_numpy()
        displacement_m = integrated[2]["acceleration"].to_numpy()

        # Step 1: Detrend velocity
        velocity_m_s = detrend(velocity_m_s, type='linear')

        # Step 2: Bandpass filter for velocity
        # Applied band pass filter using Butterworth filter
        velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
        velocity_df = butterworth(velocity_df, low_cutoff=fmin_hz, high_cutoff=fmax_hz, half_order=3)
        velocity_m_s = velocity_df['velocity'].to_numpy()

        # Step 3: Apply Hanning window before FFT
        velocity_mm_s = velocity_m_s * 1000  # Convert to mm/s
        window = windows.hann(len(velocity_mm_s))
        velocity_windowed = velocity_mm_s * window
        window_correction = np.sqrt(np.mean(window ** 2))

        # step 4: Detrend displacement  
        displacement_m = detrend(displacement_m, type='linear')

        # Step 5: Bandpass filter for displacement
        displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
        displacement_df = butterworth(displacement_df, low_cutoff=fmin_hz, high_cutoff=fmax_hz, half_order=3)
        displacement_m_s = displacement_df['displacement'].to_numpy()

        displacement_um = displacement_m_s * 1e6  # Convert to µm
        window_d = windows.hann(len(displacement_um))
        displacement_windowed = displacement_um * window_d
        window_correction_d = np.sqrt(np.mean(window_d ** 2))


        

        N1 = len(velocity_mm_s)
        N2 = len(displacement_um)
       

        # RMS and metrics
        vel_rms = rms(pd.Series(velocity_mm_s))
        disp_pp = np.ptp(displacement_um)
        acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
        acc_peak = np.max(np.abs(acceleration_g))

        # FFT for acceleration
        fft_result = fft(acceleration_g)
        freqs = fftfreq(N, T)
        fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
        pos_freqs = freqs[:N // 2]
        dom_freq = pos_freqs[np.argmax(fft_mags)]

        # FFT for velocity (with windowing)
        fft_result_vel = fft(velocity_windowed)
        freqs_vel = fftfreq(N1, T)
        fft_mags_vel = (2.0 / N1) * np.abs(fft_result_vel[:N1 // 2])
        pos_freqs_vel = freqs_vel[:N1 // 2]
        #dom_freq_vel = pos_freqs_vel[np.argmax(fft_mags_vel)]

        #FFT for displacement (with windowing)
        fft_result_disp = fft(displacement_windowed)
        freqs_disp = fftfreq(N2, T)
        fft_mags_disp = (2.0 / N2) * np.abs(fft_result_disp[:N2 // 2])
        pos_freqs_disp = freqs_disp[:N2 // 2]
        #dom_freq_disp = pos_freqs_disp[np.argmax(fft_mags_disp)]


        # Normalize RMS FFT for windowing loss
        rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / window_correction
        #velocity peak vaalue
        vel_peak = np.max(np.abs(velocity_mm_s))
        print("Velocity Peak (mm/s):", vel_peak)
        vel_peak1= np.max(np.abs(velocity_windowed))
        print("Velocity Peak (mm/s) with windowing:", vel_peak1)
        # displacement peak value
        disp_peak = np.max(np.abs(displacement_um))
        disp_peak1 = np.max(np.abs(displacement_windowed))
        print("Displacement Peak (um):", disp_peak)
        print("Displacement Peak (um) with windowing:", disp_peak1)
        print("Displacement Peak (um):", disp_pp)

        print("Velocity RMS (mm/s):", vel_rms)
        print("Dominant Frequency (Hz):", dom_freq)
        print("RMS FFT:", rms_fft)

        return {
            "acceleration": acceleration_g,
            "velocity": velocity_mm_s,
            "displacement": displacement_um,
            "time": np.linspace(0, N * T, N, endpoint=False),
            "acc_peak": acc_peak,
            "acc_rms": acc_rms,
            "vel_rms": vel_rms,
            "disp_pp": disp_pp,
            "dom_freq": dom_freq,
            "fft_freqs": pos_freqs,
            "fft_mags": fft_mags,
            "freqs_vel": pos_freqs_vel,
            "fft_mags_vel": fft_mags_vel,
            "fft_freqs_disp": pos_freqs_disp,
            "fft_mags_disp": fft_mags_disp,
            #"dom_freq_vel": dom_freq_vel,
            "rms_fft": rms_fft
        }

    def get_latest_waveform(self,fmax_hz=500,fmin_hz=1):
        ch0_voltage, ch1_voltage = self.read_data()

        print(f"\n🔍 Analyzing Channel 0")
        result_ch0 = self.analyze(ch0_voltage,fmax_hz=fmax_hz,fmin_hz=fmin_hz) if ch0_voltage.size > 0 else self._empty_result()

        print(f"\n🔍 Analyzing Channel 1")
        result_ch1 = self.analyze(ch1_voltage,fmax_hz=fmax_hz,fmin_hz=fmin_hz) if ch1_voltage.size > 0 else self._empty_result()

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