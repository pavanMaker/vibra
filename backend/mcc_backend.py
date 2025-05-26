import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, windows
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms


class Mcc172Backend:
    def __init__(self, board_num=0, sample_rate=11200, sensitivity=0.1, channel=None):
        self.board_num = board_num
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        #self.channel = channel if channel is not None else self.auto_detect_channel()
        self.board = mcc172(board_num)
        self.buffer_size = None
        self.actual_rate = None

    # def auto_detect_channel(self):
    #     for ch in [0, 1]:
    #         self.board.iepe_config_write(ch, 1)
    #         self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
    #         _, actual_rate, _ = self.board.a_in_clock_config_read()
    #         buffer_size = 2 ** int(np.floor(np.log2(actual_rate * 10)))
    #         self.board.a_in_scan_start(1 << ch, buffer_size, OptionFlags.CONTINUOUS)
    #         ch0_result,ch1_result = self.board.a_in_scan_read_numpy(-1, timeout=5.0)
    #         self.board.a_in_scan_stop()
    #         if ch0_result,ch1_result and np.any(ch0_result,ch1_result.data):
    #             print(f"IEPE is connected to channel {ch}")
    #             return ch
    #     print("No IEPE sensor is connected")
    #     return 0

    def setup(self):
        self.board.iepe_config_write(0, 1)
        self.board.iepe_config_write(1, 1)
        self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
        _, self.actual_rate, _ = self.board.a_in_clock_config_read()

        self.buffer_size = 65536
        print(f"Actual sample rate: {self.actual_rate} Hz and buffer size: {self.buffer_size}")



    def start_acquisition(self):
        channel_mask = (1<< 0) | (1<<1)#e both channels 0 and 1
        self.board.a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)

    def stop_scan(self):
        self.board.a_in_scan_stop()
        self.board.a_in_scan_cleanup()
        print("Scan stopped and cleaned up.")

    # def read_data(self):
    #     try:
    #         result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
    #     except Exception as e:
    #         print(f"Error reading data: {e}")
    #         return np.zeros(self.buffer_size), np.zeros(self.buffer_size)
    #     if  result and result.data.size:
    #         data = result.data
    #         if data.ndim == 2 and data.shape[1] == 2:
    #             ch0  = data[:, 0]
    #             ch1  = data[:, 1]
    #             return ch0,ch1
            
    #         else:
    #             print("Data format is not as expected. Expected two channels.")

    #     return np.zeros(self.buffer_size), np.zeros(self.buffer_size)  # Return empty arrays if no data
    #             # If data has two channels, average them

    def read_data(self):
        try:
            result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
        except Exception as e:
            print(f"Error reading data: {e}")
            return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

        if result and result.data.size:
            data = result.data

            # Handle dual-channel properly
            if data.ndim == 2:
                ch0 = data[:, 0] if data.shape[1] > 0 else np.zeros(self.buffer_size)
                ch1 = data[:, 1] if data.shape[1] > 1 else np.zeros(self.buffer_size)

                # Simple signal presence detection: check if channel has non-zero variance
                ch0_valid = not np.allclose(ch0, 0, atol=1e-4)
                ch1_valid = not np.allclose(ch1, 0, atol=1e-4)

                if ch0_valid and not ch1_valid:
                    print("Sensor detected on Channel 0.")
                    return ch0, np.zeros_like(ch0)
                elif ch1_valid and not ch0_valid:
                    print("Sensor detected on Channel 1.")
                    return np.zeros_like(ch1), ch1
                elif ch0_valid and ch1_valid:
                    print("Sensors detected on both channels.")
                    return ch0, ch1
                else:
                    print("No sensor signal detected.")
                    return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

            elif data.ndim == 1:
                print("Only one channel of data returned; assuming it's Channel 0.")
                return data, np.zeros_like(data)

        print("No valid data received.")
        return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

            
           

    # def analyze(self,signal_data,fmax=3600):
    #     # Convert voltage to acceleration in g
    #     acceleration_g = signal_data/self.sensitivity
    #     acceleration_g = detrend(acceleration_g, type='linear')  # Remove trend
    #     acceleration_m_s2 = acceleration_g * 9.80665  # Convert to m/s²

    #     # Time setup
    #     N = len(acceleration_m_s2)
    #     T = 1 / self.actual_rate
    #     time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
    #     df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])

    #     # Integrate to velocity and displacement
    #     integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)
    #     velocity_m_s = integrated[1]["acceleration"].to_numpy()
    #     displacement_m = integrated[2]["acceleration"].to_numpy()

    #     # Step 1: Detrend velocity
    #     velocity_m_s = detrend(velocity_m_s, type='linear')

    #     print("fmax:", fmax)

    #     # Step 2: Bandpass filter for velocity
    #     # Applied band pass filter using Butterworth filter
    #     velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
    #     velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
    #     velocity_m_s = velocity_df['velocity'].to_numpy()

    #     # Step 3: Apply Hanning window before FFT
    #     velocity_mm_s = velocity_m_s * 1000  # Convert to mm/s
    #     window = windows.hann(len(velocity_mm_s))
    #     velocity_windowed = velocity_mm_s * window
    #     window_correction = np.sqrt(np.mean(window ** 2))

    #     # step 4: Detrend displacement  
    #     displacement_m = detrend(displacement_m, type='linear')

    #     # Step 5: Bandpass filter for displacement
    #     displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
    #     displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
    #     displacement_m_s = displacement_df['displacement'].to_numpy()

    #     displacement_um = displacement_m_s * 1e6  # Convert to µm
    #     window_d = windows.hann(len(displacement_um))
    #     displacement_windowed = displacement_um * window_d
    #     window_correction_d = np.sqrt(np.mean(window_d ** 2))


        

    #     N1 = len(velocity_mm_s)
    #     N2 = len(displacement_um)
       

    #     # RMS and metrics
    #     vel_rms = rms(pd.Series(velocity_mm_s))
    #     disp_pp = np.ptp(displacement_um)
    #     acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
    #     acc_peak = np.max(np.abs(acceleration_g))

    #     # FFT for acceleration
    #     fft_result = fft(acceleration_g)
    #     freqs = fftfreq(N, T)
    #     fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
    #     pos_freqs = freqs[:N // 2]
    #     dom_freq = pos_freqs[np.argmax(fft_mags)]

    #     # FFT for velocity (with windowing)
    #     fft_result_vel = fft(velocity_windowed)
    #     freqs_vel = fftfreq(N1, T)
    #     fft_mags_vel = (2.0 / N1) * np.abs(fft_result_vel[:N1 // 2])
    #     pos_freqs_vel = freqs_vel[:N1 // 2]
    #     #dom_freq_vel = pos_freqs_vel[np.argmax(fft_mags_vel)]

    #     #FFT for displacement (with windowing)
    #     fft_result_disp = fft(displacement_windowed)
    #     freqs_disp = fftfreq(N2, T)
    #     fft_mags_disp = (2.0 / N2) * np.abs(fft_result_disp[:N2 // 2])
    #     pos_freqs_disp = freqs_disp[:N2 // 2]
    #     #dom_freq_disp = pos_freqs_disp[np.argmax(fft_mags_disp)]


    #     # Normalize RMS FFT for windowing loss
    #     rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / window_correction
    #     #velocity peak vaalue
    #     vel_peak = np.max(np.abs(velocity_mm_s))
    #     print("Velocity Peak (mm/s):", vel_peak)
    #     vel_peak1= np.max(np.abs(velocity_windowed))
    #     print("Velocity Peak (mm/s) with windowing:", vel_peak1)
    #     # displacement peak value
    #     disp_peak = np.max(np.abs(displacement_um))
    #     disp_peak1 = np.max(np.abs(displacement_windowed))
    #     print("Displacement Peak (um):", disp_peak)
    #     print("Displacement Peak (um) with windowing:", disp_peak1)
    #     print("Displacement Peak (um):", disp_pp)

    #     print("Velocity RMS (mm/s):", vel_rms)
    #     print("Dominant Frequency (Hz):", dom_freq)
    #     print("RMS FFT:", rms_fft)

    #     return {
    #         "acceleration": acceleration_g,
    #         "velocity": velocity_mm_s,
    #         "displacement": displacement_um,
    #         "time": np.linspace(0, N * T, N, endpoint=False),
    #         "acc_peak": acc_peak,
    #         "acc_rms": acc_rms,
    #         "vel_rms": vel_rms,
    #         "disp_pp": disp_pp,
    #         "dom_freq": dom_freq,
    #         "fft_freqs": pos_freqs,
    #         "fft_mags": fft_mags,
    #         "freqs_vel": pos_freqs_vel,
    #         "fft_mags_vel": fft_mags_vel,
    #         "fft_freqs_disp": pos_freqs_disp,
    #         "fft_mags_disp": fft_mags_disp,
    #         #"dom_freq_vel": dom_freq_vel,
    #         "rms_fft": rms_fft
    #     }

    def analyze(self, result_data,fmax=3600):
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
        velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=500, half_order=3)
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
        displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=500, half_order=3)
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


    def get_latest_waveform(self,fmax=3600):
        ch0_data,ch1_data = self.read_data()
        if len(ch0_data) == 0:
            print("No data received from MCC 172.")
            return [], [], [], [], 0, 0, 0, 0, 0

        ch0_result = self.analyze(ch0_data,fmax=fmax)
        ch1_result=self.analyze(ch1_data,fmax=fmax)
       
        return ch0_result,ch1_result



# import numpy as np
# import pandas as pd
# from daqhats import mcc172, OptionFlags, SourceType
# from scipy.fft import fft, fftfreq
# from scipy.signal import detrend, windows
# from endaq.calc.integrate import integrals
# # from endaq.calc.filters import butterworth  # Temporarily bypassed
# from endaq.calc.stats import rms

# class Mcc172Backend:
#     def __init__(self, board_num=0, sample_rate=11200, sensitivity=0.1):
#         self.board_num = board_num
#         self.sample_rate = sample_rate
#         self.sensitivity = sensitivity
#         self.board = mcc172(board_num)
#         self.buffer_size = None
#         self.actual_rate = None

#     def setup(self):
#         self.board.iepe_config_write(0, 1)
#         self.board.iepe_config_write(1, 1)
#         self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
#         _, self.actual_rate, _ = self.board.a_in_clock_config_read()
#         self.buffer_size =65536
#         print(f"[Setup] Actual Sample Rate: {self.actual_rate} Hz, Buffer Size: {self.buffer_size}")

#     def start_acquisition(self):
#         channel_mask = (1 << 0) | (1 << 1)
#         self.board.a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)

#     def stop_scan(self):
#         self.board.a_in_scan_stop()
#         self.board.a_in_scan_cleanup()
#         print("[Stop] Scan stopped and cleaned up.")

#     def read_data(self):
#         try:
#             result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
#         except Exception as e:
#             print(f"[Read Error] {e}")
#             return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

#         if result and result.data.size:
#             data = result.data
#             print(f"[Read] Raw shape: {data.shape}")
#             if data.ndim == 2 and data.shape[1] == 2:
#                 ch0 = data[:, 0]
#                 ch1 = data[:, 1]
#                 print(f"[Ch0] Mean: {np.mean(ch0):.4f}, Std: {np.std(ch0):.4f}")
#                 print(f"[Ch1] Mean: {np.mean(ch1):.4f}, Std: {np.std(ch1):.4f}")
#                 return ch0, ch1
#             elif data.ndim == 1:
#                 print("[Read] Only 1 channel active, assigning second channel to zero.")
#                 return data, np.zeros_like(data)

#         print("[Read] No valid data received.")
#         return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

#     def analyze(self, signal_data, fmax=3600):
#         acceleration_g = signal_data / self.sensitivity
#         acceleration_g = detrend(acceleration_g, type='linear')
#         print("[Analyze] First 10 Acc (g):", acceleration_g[:10])

#         acceleration_m_s2 = acceleration_g * 9.80665
#         N = len(acceleration_m_s2)
#         T = 1 / self.actual_rate
#         time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
#         df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])

#         integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)
#         velocity_m_s = integrated[1]["acceleration"].to_numpy()
#         displacement_m = integrated[2]["acceleration"].to_numpy()

#         velocity_m_s = detrend(velocity_m_s, type='linear')
#         # Bypassing filters for debug
#         # velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
#         # velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
#         # velocity_m_s = velocity_df['velocity'].to_numpy()

#         velocity_mm_s = velocity_m_s * 1000
#         window = windows.hann(len(velocity_mm_s))
#         velocity_windowed = velocity_mm_s * window
#         window_correction = np.sqrt(np.mean(window ** 2))

#         displacement_m = detrend(displacement_m, type='linear')
#         # displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
#         # displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
#         # displacement_m = displacement_df['displacement'].to_numpy()

#         displacement_um = displacement_m * 1e6
#         window_d = windows.hann(len(displacement_um))
#         displacement_windowed = displacement_um * window_d

#         N1 = len(velocity_mm_s)
#         N2 = len(displacement_um)

#         # RMS and Metrics
#         vel_rms = rms(pd.Series(velocity_mm_s))
#         disp_pp = np.ptp(displacement_um)
#         acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
#         acc_peak = np.max(np.abs(acceleration_g))

#         # FFT
#         fft_result = fft(acceleration_g)
#         freqs = fftfreq(N, T)
#         fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
#         pos_freqs = freqs[:N // 2]
#         dom_freq = pos_freqs[np.argmax(fft_mags)]

#         fft_result_vel = fft(velocity_windowed)
#         freqs_vel = fftfreq(N1, T)
#         fft_mags_vel = (2.0 / N1) * np.abs(fft_result_vel[:N1 // 2])
#         pos_freqs_vel = freqs_vel[:N1 // 2]

#         fft_result_disp = fft(displacement_windowed)
#         freqs_disp = fftfreq(N2, T)
#         fft_mags_disp = (2.0 / N2) * np.abs(fft_result_disp[:N2 // 2])
#         pos_freqs_disp = freqs_disp[:N2 // 2]

#         rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / window_correction

#         print(f"[Analyze] Peak Acc (g): {acc_peak:.2f}, RMS Vel (mm/s): {vel_rms:.2f}, Disp (pp): {disp_pp:.2f}")
#         print(f"[Analyze] Dom Freq: {dom_freq:.2f} Hz")

#         return {
#             "acceleration": acceleration_g,
#             "velocity": velocity_mm_s,
#             "displacement": displacement_um,
#             "time": np.linspace(0, N * T, N, endpoint=False),
#             "acc_peak": acc_peak,
#             "acc_rms": acc_rms,
#             "vel_rms": vel_rms,
#             "disp_pp": disp_pp,
#             "dom_freq": dom_freq,
#             "fft_freqs": pos_freqs,
#             "fft_mags": fft_mags,
#             "freqs_vel": pos_freqs_vel,
#             "fft_mags_vel": fft_mags_vel,
#             "fft_freqs_disp": pos_freqs_disp,
#             "fft_mags_disp": fft_mags_disp,
#             "rms_fft": rms_fft
#         }

#     def get_latest_waveform(self, fmax=3600):
#         ch0_data, ch1_data = self.read_data()
#         if len(ch0_data) == 0:
#             print("[Waveform] No data from Ch0.")
#             return {}, {}

#         ch0_result = self.analyze(ch0_data, fmax=fmax)
#         ch1_result = self.analyze(ch1_data, fmax=fmax)
#         return ch0_result, ch1_result


