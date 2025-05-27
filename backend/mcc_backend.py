import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, windows
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms

class Mcc172Backend:
    def __init__(self, board_num=0, sample_rate=11200, sensitivity=0.1):
        self.board_num = board_num
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        self.board = mcc172(board_num)
        self.buffer_size = None
        self.actual_rate = None
        self.last_active_channel = None

    def setup(self):
        for ch in [0, 1]:
            self.board.iepe_config_write(ch, 1)
        self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
        _, self.actual_rate, _ = self.board.a_in_clock_config_read()
        self.buffer_size = 65536
        print(f"Actual sample rate: {self.actual_rate} Hz and buffer size: {self.buffer_size}")

    def start_acquisition(self):
        self.board.a_in_scan_start((1 << 0) | (1 << 1), self.buffer_size, OptionFlags.CONTINUOUS)

    def stop_scan(self):
        self.board.a_in_scan_stop()
        self.board.a_in_scan_cleanup()
        print("Scan stopped and cleaned up.")

    # def read_data(self):
    #     try:
    #         result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
    #     except Exception as e:
    #         print(f"Read error: {e}")
    #         return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

    #     if result and result.data.size:
    #         data = result.data
    #         if data.ndim == 2 and data.shape[1] == 2:
    #             return data[:, 0], data[:, 1]
    #         elif data.ndim == 1:
    #             print("Only one channel returned, splitting equally")
    #             half = len(data) // 2
    #             return data[:half], np.zeros(half)
    #         else:
    #             print("Unexpected data shape:", data.shape)

    #     return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

    def read_data(self):
        try:
            result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
        except Exception as e:
            print(f"[ERROR] Read error: {e}")
            return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

        if result and result.data.size:
            data = result.data

            # ✅ Case 1: both channels returned
            if data.ndim == 2 and data.shape[1] == 2:
                ch0 = data[:, 0]
                ch1 = data[:, 1]
                ch0_rms = np.sqrt(np.mean(ch0**2))
                ch1_rms = np.sqrt(np.mean(ch1**2))
                print(f"[DEBUG] Ch0 RMS: {ch0_rms:.5f}, Ch1 RMS: {ch1_rms:.5f}")
                return ch0, ch1

            # ✅ Case 2: only one channel returned (1D array)
            elif data.ndim == 1:
                rms = np.sqrt(np.mean(data**2))
                print(f"[DEBUG] Only one channel returned. RMS: {rms:.5f}")

                # If signal is strong, decide based on content
                if rms > 0.01:
                    # Use first and second halves to infer which channel
                    half = len(data) // 2
                    first_half_rms = np.sqrt(np.mean(data[:half]**2))
                    second_half_rms = np.sqrt(np.mean(data[half:]**2))

                    print(f"[DEBUG] First half RMS: {first_half_rms:.5f}, Second half RMS: {second_half_rms:.5f}")

                    # Heuristic: Assume Ch0 if signal is in first half (DAQHats returns Ch0 first)
                    if first_half_rms > second_half_rms:
                        return data[:half], np.zeros_like(data[:half])  # Signal on Ch0
                    else:
                        return np.zeros_like(data[:half]), data[:half]  # Signal on Ch1
                else:
                    # Too low — no signal
                    print("[INFO] Signal too weak. Returning zeros.")
                    return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

            # ❌ Unexpected shape
            else:
                print(f"[ERROR] Unexpected data shape: {data.shape}")
                return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

        print("[ERROR] No data received.")
        return np.zeros(self.buffer_size), np.zeros(self.buffer_size)



    def analyze(self, signal_data, fmax=3600):
        acceleration_g = signal_data / self.sensitivity
        acceleration_g = detrend(acceleration_g, type='linear')
        acceleration_m_s2 = acceleration_g * 9.80665

        N = len(acceleration_m_s2)
        T = 1 / self.actual_rate
        time_index = np.linspace(0, N * T, N, endpoint=False)

        df = pd.DataFrame(acceleration_m_s2, index=pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T)), columns=["acceleration"])
        integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)

        velocity_m_s = detrend(integrated[1]["acceleration"].to_numpy(), type='linear')
        displacement_m = detrend(integrated[2]["acceleration"].to_numpy(), type='linear')

        # Bandpass filter
        velocity_df = butterworth(pd.DataFrame(velocity_m_s, columns=["velocity"], index=df.index),
                                  low_cutoff=3, high_cutoff=fmax, half_order=3)
        displacement_df = butterworth(pd.DataFrame(displacement_m, columns=["displacement"], index=df.index),
                                      low_cutoff=3, high_cutoff=fmax, half_order=3)

        velocity_mm_s = velocity_df["velocity"].to_numpy() * 1000
        displacement_um = displacement_df["displacement"].to_numpy() * 1e6

        # FFT setup
        window_v = windows.hann(N)
        velocity_windowed = velocity_mm_s * window_v
        fft_vel = fft(velocity_windowed)
        fft_freqs_vel = fftfreq(N, T)[:N // 2]
        fft_mags_vel = (2.0 / N) * np.abs(fft_vel[:N // 2])

        window_d = windows.hann(N)
        displacement_windowed = displacement_um * window_d
        fft_disp = fft(displacement_windowed)
        fft_freqs_disp = fftfreq(N, T)[:N // 2]
        fft_mags_disp = (2.0 / N) * np.abs(fft_disp[:N // 2])

        fft_acc = fft(acceleration_g)
        fft_freqs_acc = fftfreq(N, T)[:N // 2]
        fft_mags_acc = (2.0 / N) * np.abs(fft_acc[:N // 2])

        dom_freq = fft_freqs_acc[np.argmax(fft_mags_acc)]
        vel_rms = rms(pd.Series(velocity_mm_s))
        disp_pp = np.ptp(displacement_um)
        acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
        acc_peak = np.max(np.abs(acceleration_g))
        rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / np.sqrt(np.mean(window_v ** 2))

        return {
            "acceleration": acceleration_g,
            "velocity": velocity_mm_s,
            "displacement": displacement_um,
            "time": time_index,
            "acc_peak": acc_peak,
            "acc_rms": acc_rms,
            "vel_rms": vel_rms,
            "disp_pp": disp_pp,
            "dom_freq": dom_freq,
            "fft_freqs": fft_freqs_acc,
            "fft_mags": fft_mags_acc,
            "freqs_vel": fft_freqs_vel,
            "fft_mags_vel": fft_mags_vel,
            "fft_freqs_disp": fft_freqs_disp,
            "fft_mags_disp": fft_mags_disp,
            "rms_fft": rms_fft
        }
    def get_latest_waveform(self, fmax=3600):
        ch0_data, ch1_data = self.read_data()

        ch0_rms = np.sqrt(np.mean(ch0_data**2))
        ch1_rms = np.sqrt(np.mean(ch1_data**2))
        ch0_valid = ch0_rms > 0.01
        ch1_valid = ch1_rms > 0.01

        print(f"[DEBUG] get_latest_waveform() -> Ch0 RMS: {ch0_rms:.4f}, Ch1 RMS: {ch1_rms:.4f}")
        print(f"[DEBUG] Valid channels -> Ch0: {ch0_valid}, Ch1: {ch1_valid}")

        if ch0_valid and not ch1_valid:
            active_channel = 0
        elif ch1_valid and not ch0_valid:
            active_channel = 1
        elif ch0_valid and ch1_valid:
            active_channel = -1  # both active
        else:
            active_channel = -1  # neither active

        ch0_result = self.analyze(ch0_data, fmax=fmax) if ch0_valid else self.analyze(np.zeros_like(ch0_data), fmax=fmax)
        ch1_result = self.analyze(ch1_data, fmax=fmax) if ch1_valid else self.analyze(np.zeros_like(ch1_data), fmax=fmax)

        return ch0_result, ch1_result, active_channel


    # def get_latest_waveform(self, fmax=3600):
    #     ch0_data, ch1_data = self.read_data()

    #     ch0_valid = not np.allclose(ch0_data, 0, atol=1e-4)
    #     ch1_valid = not np.allclose(ch1_data, 0, atol=1e-4)

    #     if ch0_valid:
    #         ch0_result = self.analyze(ch0_data, fmax=fmax)
    #     else:
    #         ch0_result = self.analyze(np.zeros_like(ch0_data), fmax=fmax)

    #     if ch1_valid:
    #         ch1_result = self.analyze(ch1_data, fmax=fmax)
    #     else:
    #         ch1_result = self.analyze(np.zeros_like(ch1_data), fmax=fmax)

    #     return ch0_result, ch1_result

    
