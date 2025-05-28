import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, windows
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms


class Mcc172Backend:
    def __init__(self, enabled_channels, sample_rate=51200):
        self.enabled_channels = enabled_channels  # List of dicts: board_num, channel, sensitivity
        self.sample_rate = sample_rate
        self.boards = {}  # board_num -> mcc172 instance
        self.buffer_size = 2 ** 16  # 65536
        self.actual_rate = None

    def setup(self):
        for config in self.enabled_channels:
            bnum = config['board_num']
            ch = config['channel']
            if bnum not in self.boards:
                self.boards[bnum] = mcc172(bnum)
                self.boards[bnum].a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
                _, self.actual_rate, _ = self.boards[bnum].a_in_clock_config_read()
            self.boards[bnum].iepe_config_write(ch, 1)

    def start_acquisition(self):
        for bnum, board in self.boards.items():
            channels = [cfg['channel'] for cfg in self.enabled_channels if cfg['board_num'] == bnum]
            ch_mask = sum([1 << ch for ch in channels])
            board.a_in_scan_start(ch_mask, self.buffer_size, OptionFlags.CONTINUOUS)

    def stop_scan(self):
        for board in self.boards.values():
            board.a_in_scan_stop()
            board.a_in_scan_cleanup()

    def read_data(self):
        channel_data = []

        for config in self.enabled_channels:
            bnum = config['board_num']
            ch = config['channel']
            sens = config['sensitivity']

            result = self.boards[bnum].a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
            if result and result.data is not None:
                try:
                    if result.data.ndim == 1:
                        voltage = result.data
                    else:
                        voltage = result.data[ch]
                    channel_data.append({
                        "board_num": bnum,
                        "channel": ch,
                        "sensitivity": sens,
                        "voltage": voltage
                    })
                except Exception as e:
                    print(f"Error reading channel {ch} from board {bnum}: {e}")
                    channel_data.append({
                        "board_num": bnum,
                        "channel": ch,
                        "sensitivity": sens,
                        "voltage": np.zeros(self.buffer_size)
                    })
            else:
                print(f"No data from board {bnum}, channel {ch}")
                channel_data.append({
                    "board_num": bnum,
                    "channel": ch,
                    "sensitivity": sens,
                    "voltage": np.zeros(self.buffer_size)
                })

        return channel_data

    def empty_result(self):
        return {
            "time": [],
            "acceleration": [],
            "velocity": [],
            "displacement": [],
            "acc_peak": 0,
            "acc_rms": 0,
            "vel_rms": 0,
            "disp_pp": 0,
            "dom_freq": 0,
            "fft_freqs": [],
            "fft_mags": [],
            "fft_mags_vel": [],
            "fft_mags_disp": [],
            "rms_fft": 0
        }

    def analyze(self, voltage_array, sensitivity):
        if voltage_array is None or len(voltage_array) == 0:
            print("⚠️ Empty voltage array received in analyze()")
            return self.empty_result()

        acceleration_g = voltage_array / sensitivity
        acceleration_g = detrend(acceleration_g, type='linear')
        acceleration_m_s2 = acceleration_g * 9.80665

        N = len(acceleration_m_s2)
        T = 1 / self.actual_rate
        time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
        df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])

        integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)
        velocity_m_s = detrend(integrated[1]["acceleration"].to_numpy(), type='linear')
        displacement_m = detrend(integrated[2]["acceleration"].to_numpy(), type='linear')

        velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
        velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=500, half_order=3)
        velocity_mm_s = velocity_df['velocity'].to_numpy() * 1000

        displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
        displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=500, half_order=3)
        displacement_um = displacement_df['displacement'].to_numpy() * 1e6

        window = windows.hann(N)
        fft_acc = fft(acceleration_g * window)
        fft_vel = fft(velocity_mm_s * window)
        fft_disp = fft(displacement_um * window)

        freqs = fftfreq(N, T)
        pos_mask = freqs[:N // 2] <= 500

        fft_mags_acc = (2.0 / N) * np.abs(fft_acc[:N // 2])
        fft_mags_vel = (2.0 / N) * np.abs(fft_vel[:N // 2])
        fft_mags_disp = (2.0 / N) * np.abs(fft_disp[:N // 2])

        dom_freq = freqs[:N // 2][np.argmax(fft_mags_acc)]

        return {
            "time": np.linspace(0, N * T, N, endpoint=False),
            "acceleration": acceleration_g,
            "velocity": velocity_mm_s,
            "displacement": displacement_um,
            "acc_peak": np.max(np.abs(acceleration_g)),
            "acc_rms": np.sqrt(np.mean(acceleration_g ** 2)),
            "vel_rms": rms(pd.Series(velocity_mm_s)),
            "disp_pp": np.ptp(displacement_um),
            "dom_freq": dom_freq,
            "fft_freqs": freqs[:N // 2][pos_mask],
            "fft_mags": fft_mags_acc[pos_mask],
            "fft_mags_vel": fft_mags_vel[pos_mask],
            "fft_mags_disp": fft_mags_disp[pos_mask],
            "rms_fft": np.sqrt(np.sum((fft_mags_vel[pos_mask][1:] ** 2) / 2))
        }

    def get_latest_waveform(self):
        channel_data = self.read_data()
        if not channel_data:
            print("❌ No channel data available.")
            return [], [], [], [], 0, 0, 0, 0, 0

        ch = channel_data[0]
        result = self.analyze(ch["voltage"], ch["sensitivity"])

        return (
            result["time"].tolist(),
            result["acceleration"].tolist(),
            result["velocity"].tolist(),
            result["displacement"].tolist(),
            result["acc_peak"],
            result["acc_rms"],
            result["vel_rms"],
            result["disp_pp"],
            result["dom_freq"],
            result["fft_freqs"],
            result["fft_mags"],
            result["fft_freqs"],
            result["fft_mags_vel"],
            result["fft_freqs"],
            result["fft_mags_disp"],
            result["rms_fft"]
        )
