# --- File: mcc_backend.py ---

import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, windows
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms

class Mcc172Backend:
    def __init__(self, channel_configs, selected_channels, sample_rate=51200):
        self.channel_configs  =  channel_configs  # List of dicts: board_num, channel, sensitivity
        self.selected_channels = selected_channels
        self.sample_rate = sample_rate
        self.boards = {}  # board_num -> mcc172 instance
        self.buffer_size = 2 ** 16  # 65536
        self.actual_rate = None
        self.active_configs = []

    def setup(self):
        self.active_configs = []
        ch_config_map = {cfg['board_num'] * 2 + cfg['channel']: cfg for cfg in self.channel_configs}
        for ch_index in self.selected_channels:
            cfg = ch_config_map.get(ch_index)
            if not cfg:
                print(f" Channel {ch_index} not found in configuration.")
                continue

            try:
                board_num = cfg['board_num']
                channel = cfg['channel']
                sensitivity = cfg['sensitivity']
                self.active_configs.append(cfg)

                if board_num not in self.boards:
                    self.boards[board_num] = mcc172(board_num)
                    self.boards[board_num].a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
                    _, self.actual_rate,_ = self.boards[board_num].a_in_clock_config_read()
                    print(f"Board {board_num} configured with sample rate: {self.actual_rate} Hz")

                self.boards[board_num].iepe_config_write(channel, 1)
            except Exception as e:
                print(f"\u274c Error configuring channel {ch_index}: {e}")

    def start_acquisition(self):
        board_channel_map = {}
        for config in self.active_configs:
            bnum = config['board_num']
            ch = config['channel']
            board_channel_map.setdefault(bnum, []).append(ch)

        for bnum, channels in board_channel_map.items():
            ch_mask = sum(1 << ch for ch in channels)
            try:
                self.boards[bnum].a_in_scan_start(ch_mask, self.buffer_size, self.sample_rate, OptionFlags.DEFAULT)
                print(f"✅ Started acquisition on board {bnum} for channels {channels}")
            except Exception as e:
                print(f"❌ Error starting acquisition on board {bnum}: {e}")



    def stop_scan(self):
        for board in self.boards.values():
            board.a_in_scan_stop()
            board.a_in_scan_cleanup()

    def read_data(self):
        channel_data = []
        for config in self.active_configs:
            bnum = config['board_num']
            ch = config['channel']
            sens = config['sensitivity']
            try:
                result = self.boards[bnum].a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
                if result and result.data is not None:
                    if result.data.ndim > 1:
                        print(f"📦 Got shape: {result.data.shape} — using board {bnum}, channel {ch}")
                        voltage = result.data[ch]
                    else:
                        print(f"📦 Got 1D data — using board {bnum}, channel {ch}")
                        voltage = result.data
                    print(f"Read data from board {bnum}, channel {ch}: {voltage[:10]}...")  # show first 10 values
                    channel_data.append({
                        "board_num": bnum,
                        "channel": ch,
                        "sensitivity": sens,
                        "voltage": voltage
                    })
                else:
                    print(f"⚠️ No data in buffer for board {bnum}, channel {ch}")
                    channel_data.append({
                        "board_num": bnum,
                        "channel": ch,
                        "sensitivity": sens,
                        "voltage": np.zeros(self.buffer_size)
                    })
            except Exception as e:
                print(f"❌ Error reading board {bnum}, channel {ch}: {e}")
                channel_data.append({
                    "board_num": bnum,
                    "channel": ch,
                    "sensitivity": sens,
                    "voltage": np.zeros(self.buffer_size)
                })
                print(f"⚠️ Using zeroed data for board {bnum}, channel {ch} due to error.")
            print(f"✅ Channel {ch} data length: {len(channel_data[-1]['voltage'])}")  # confirm array length
        return channel_data




    def analyze(self, voltage_array, sensitivity):
        acceleration_g = voltage_array / sensitivity
        print("acceleration_g:", acceleration_g[:10])  # Debugging line to check initial values
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
        window_v = windows.hann(N)
        velocity_windowed = velocity_mm_s * window_v
        correction_v = np.sqrt(np.mean(window_v ** 2))

        displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
        displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=500, half_order=3)
        displacement_um = displacement_df['displacement'].to_numpy() * 1e6
        window_d = windows.hann(N)
        displacement_windowed = displacement_um * window_d
        correction_d = np.sqrt(np.mean(window_d ** 2))

        fft_acc = fft(acceleration_g * windows.hann(N))
        fft_vel = fft(velocity_windowed)
        fft_disp = fft(displacement_windowed)
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
            "rms_fft": np.sqrt(np.sum((fft_mags_vel[pos_mask][1:] ** 2) / 2)) / correction_v
        }

    def get_latest_waveform(self):
        channel_data = self.read_data()
        if not channel_data:
            return [self._empty_result(), self._empty_result()]

        results = []
        for ch in channel_data:
            try:
                result = self.analyze(ch["voltage"], ch["sensitivity"])
                results.append((
                    result["time"].tolist(),
                    result["acceleration"].tolist(),
                    result["velocity"].tolist(),
                    result["displacement"].tolist(),
                    result["acc_peak"],
                    result["acc_rms"],
                    result["vel_rms"],
                    result["disp_pp"],
                    result["dom_freq"],
                    result["fft_freqs"].tolist(),
                    result["fft_mags"].tolist(),
                    result["fft_freqs"].tolist(),
                    result["fft_mags_vel"].tolist(),
                    result["fft_freqs"].tolist(),
                    result["fft_mags_disp"].tolist(),
                    result["rms_fft"]
                ))
            except Exception as e:
                print(f"\u274c Analysis failed for board {ch['board_num']} ch {ch['channel']}: {e}")
                results.append(self._empty_result())

        while len(results) < 2:
            results.append(self._empty_result())

        return results

    def _empty_result(self):
        return ([], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, [], [], [], [], [], [], 0.0)
