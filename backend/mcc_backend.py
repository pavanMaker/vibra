import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq
from endaq.calc.integrate import integrals

class Mcc172Backend:
    def __init__(self, board_num=0, sample_rate=51200, sensitivity=0.1, channel=None):
        self.board_num = board_num
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        self.channel = channel if channel is not None else self.auto_detect_channel()
        self.board = mcc172(board_num)
        self.buffer_size = None
        self.actual_rate = None

    def auto_detect_channel(self):
        for ch in [0, 1]:
            self.board.iepe_config_write(ch, 1)
            self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
            _, actual_rate, _ = self.board.a_in_clock_config_read()
            buffer_size = 2 ** int(np.floor(np.log2(actual_rate * 10)))
            self.board.a_in_scan_start(1 << ch, buffer_size, OptionFlags.CONTINUOUS)
            result = self.board.a_in_scan_read_numpy(-1, timeout=5.0)
            self.board.a_in_scan_stop()
            if result and np.any(result.data):
                print(f"IEPE is connected to channel {ch}")
                return ch
        print("No IEPE sensor is connected")
        return 0

    def setup(self):
        self.board.iepe_config_write(self.channel, 1)
        self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
        _, self.actual_rate, _ = self.board.a_in_clock_config_read()
        self.buffer_size = 65536
        print("buffer size",self.buffer_size)

    def start_acquisition(self):
        channel_mask = 1 << self.channel
        self.board.a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)

    def stop_scan(self):
        self.board.a_in_scan_stop()

    def read_data(self):
        result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
        if result and np.any(result.data):
            print("Raw voltage data (first 10 samples):", result.data[:10])
            return result.data
        else:
            print("No data received from MCC 172 during read.")
            return np.array([])

    def analyze(self, result_data):
        # Converting voltage to acceleration in g
        acceleration_g = result_data / self.sensitivity
        acceleration_g -= np.mean(acceleration_g)

        # Time parameters
        N = len(acceleration_g)
        T = 1 / self.actual_rate
        print("No of Samples")
        print(N,T,self.actual_rate)
        

        # Creates time-indexed DataFrame
        time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
        df = pd.DataFrame(acceleration_g, index=time_index, columns=["acceleration"])

        # Integrate to velocity (m/s) and displacement (m) using Endaq
        integrated = integrals(df, n=1, highpass_cutoff=3, tukey_percent=0.05)
        velocity_m_s = integrated[1]["acceleration"]#here after integrating the g value automatically changes into m/s with this function
        displacement_m = integrated[1]["acceleration"]

        # Convert to desired units
        velocity_mm_s = velocity_m_s * 1000 # mm/s
        print("Velocity Sample (mm/s):", velocity_mm_s[:10])
        displacement_um = displacement_m * 1e6 # µm

        # RMS and peak-to-peak
        vel_rms = np.sqrt(np.mean(velocity_mm_s**2))
        disp_pp = np.ptp(displacement_um)
        acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
        acc_peak = np.max(np.abs(acceleration_g))

        # FFT for dominant frequency
        fft_result = fft(acceleration_g)#converting time domain to frequency domain
        freqs = fftfreq(N, T)#return the frequency bins 
        fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
        pos_freqs = freqs[:N // 2]
        dom_freq = pos_freqs[np.argmax(fft_mags)]

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
            "fft_mags": fft_mags
        }

    def get_latest_waveform(self):
        result_data = self.read_data()
        if len(result_data) == 0:
            print("⚠️ Empty result data from MCC 172.")
            return [], [], [], [], 0, 0, 0, 0, 0

        result = self.analyze(result_data)
        return (
            result["time"].tolist(),
            result["acceleration"].tolist(),
            result["velocity"].tolist(),
            result["displacement"].tolist(),
            result["acc_peak"],
            result["acc_rms"],
            result["vel_rms"],
            result["disp_pp"],
            result["dom_freq"]
        )
