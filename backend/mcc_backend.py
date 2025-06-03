import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq
from scipy.signal import detrend, windows
from endaq.calc.integrate import integrals
from endaq.calc.filters import butterworth
from endaq.calc.stats import rms

class Mcc172Backend:
    def __init__(self, sample_rate=51200):
        self.sample_rate = sample_rate
        self.buffer_size = 65536
        self.actual_rate = None
        self.active_configs = []
        self.boards = {}

    def set_active_channels(self, configs):
        self.active_configs = configs
        print("active channels came into the backend",self.active_configs)

    def setup(self):
        configured_boards = set()
        for cfg in self.active_configs:
            bnum = cfg['board_num']
            print("bnum:",bnum)
            ch = cfg['channel']
            print("channels setting_up",ch)
            if bnum not in self.boards:
                self.boards[bnum] = mcc172(bnum)
                print("boards connected:",self.boards[bnum])
            

            board = self.boards[bnum]
            board.iepe_config_write(ch, 1)

            if bnum not in configured_boards:
                board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
                _, self.actual_rate, _ = board.a_in_clock_config_read()
                print(f"✅ Board {bnum} configured with sample rate: {self.actual_rate} Hz")
                configured_boards.add(bnum)

    def start_acquisition(self):
        for bnum in set(cfg['board_num'] for cfg in self.active_configs):
            channel_mask = 0
            for cfg in self.active_configs:
                if cfg['board_num'] == bnum:
                    channel_mask |= (1 << cfg['channel'])
            print(f"🎬 Starting acquisition on Board {bnum}, Channel Mask: {bin(channel_mask)}")
            self.boards[bnum].a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)
            print("boards and channels",self.boards)

    def stop_scan(self):
        for board in self.boards.values():
            board.a_in_scan_stop()
            board.a_in_scan_cleanup()
        print("🛑 Scans stopped and cleaned.")

    def read_data(self):
        channel_data = []
        
        for bnum in set(cfg['board_num'] for cfg in self.active_configs):
            try:
                result = self.boards[bnum].a_in_scan_read_numpy(self.start_acquisition.channel_mask,self.buffer_size,timeout =5.0)

                if result and result.data is not None:
                    print(f"📦 Raw result shape from board {bnum}: {result.data.shape}")
                    print(f"📐 Raw result ndim from board {bnum}: {result.data.ndim}")

                    active_chs = [cfg['channel'] for cfg in self.active_configs if cfg['board_num'] == bnum]
                    print(f"📌 Active channels for board {bnum}: {active_chs}")

                    if result.data.ndim == 2:
                        if result.data.shape[0] >= 1:
                            voltage_ch0 = result.data[0]
                            print(f"🔎 voltage_ch0 (first 5): {voltage_ch0[:5]}")

                        if result.data.shape[0] >= 2:
                            voltage_ch1 = result.data[1]
                            print(f"🔎 voltage_ch1 (first 5): {voltage_ch1[:5]}")

                    elif result.data.ndim == 1:
                        voltage_ch0 = result.data
                        print(f"🔎 Single-channel voltage (first 5): {voltage_ch0[:5]}")

                    for idx, ch in enumerate(active_chs):
                        sens = next((cfg['sensitivity'] for cfg in self.active_configs
                                    if cfg['board_num'] == bnum and cfg['channel'] == ch), None)
                        if sens is None:
                            print(f"⚠️ Sensitivity missing for board {bnum} channel {ch}, using fallback 0.1")
                            sens = 0.1

                        if result.data.ndim > 1:
                            voltage = result.data[idx]
                        else:
                            voltage = result.data

                        print(f"📊 Board {bnum} CH{ch} Voltage Sample: {voltage[:5]}")
                        channel_data.append({
                            "board_num": bnum,
                            "channel": ch,
                            "sensitivity": sens,
                            "voltage": voltage
                        })

                else:
                    print(f"⚠️ No data from board {bnum}")
                    for cfg in self.active_configs:
                        if cfg['board_num'] == bnum:
                            channel_data.append({
                                "board_num": bnum,
                                "channel": cfg['channel'],
                                "sensitivity": cfg['sensitivity'],
                                "voltage": np.zeros(self.buffer_size)
                            })

            except Exception as e:
                print(f"❌ Error reading board {bnum}: {e}")
                for cfg in self.active_configs:
                    if cfg['board_num'] == bnum:
                        channel_data.append({
                            "board_num": bnum,
                            "channel": cfg['channel'],
                            "sensitivity": cfg['sensitivity'],
                            "voltage": np.zeros(self.buffer_size)
                        })

        return channel_data




    def analyze(self, voltage, sensitivity):
        acceleration_g = detrend(voltage / sensitivity, type='linear')
        acceleration_m_s2 = acceleration_g * 9.80665

        N = len(acceleration_g)
        T = 1 / self.actual_rate
        time = np.linspace(0, N * T, N, endpoint=False)
        time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
        df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])

        integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)
        velocity_m_s = detrend(integrated[1]["acceleration"].to_numpy(), type='linear')
        displacement_m = detrend(integrated[2]["acceleration"].to_numpy(), type='linear')

        velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
        velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=500, half_order=3)
        velocity_m_s = velocity_df['velocity'].to_numpy() * 1000

        displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
        displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=500, half_order=3)
        displacement_um = displacement_df['displacement'].to_numpy() * 1e6

        window = windows.hann(N)
        v_win = velocity_m_s * window
        d_win = displacement_um * window
        win_corr = np.sqrt(np.mean(window ** 2))

        fft_acc = fft(acceleration_g)
        fft_vel = fft(v_win)
        fft_disp = fft(d_win)

        freqs = fftfreq(N, T)[:N//2]
        fft_mags_acc = (2.0 / N) * np.abs(fft_acc[:N // 2])
        fft_mags_vel = (2.0 / N) * np.abs(fft_vel[:N // 2])
        fft_mags_disp = (2.0 / N) * np.abs(fft_disp[:N // 2])

        dom_freq = freqs[np.argmax(fft_mags_acc)]
        acc_peak = np.max(np.abs(acceleration_g))
        acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
        vel_rms = rms(pd.Series(velocity_m_s))
        disp_pp = np.ptp(displacement_um)
        rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / win_corr

        return {
            "t": time,
            "accel": acceleration_g,
            "velocity": velocity_m_s,
            "displacement": displacement_um,
            "fft_freqs": freqs,
            "fft_mags": fft_mags_acc,
            "fft_freqs_vel": freqs,
            "fft_mags_vel": fft_mags_vel,
            "fft_freqs_disp": freqs,
            "fft_mags_disp": fft_mags_disp,
            "acc_peak": acc_peak,
            "acc_rms": acc_rms,
            "vel_rms": vel_rms,
            "disp_pp": disp_pp,
            "dom_freq": dom_freq,
            "rms_fft": rms_fft,
        }

    def get_latest_waveform(self):
        all_results = []

        channel_data = self.read_data()  
        for ch in channel_data:
            voltage = ch["voltage"]
            sensitivity = ch["sensitivity"]
            # === PATCH: skip empty data arrays ===
            if voltage is None or len(voltage) == 0:
                print(f"⚠️ No data in board {ch['board_num']} channel {ch['channel']} - skipping analysis.")
                continue
            analyzed = self.analyze(voltage, sensitivity)
            if analyzed is not None:
                all_results.append(analyzed)

        print(f"Number of channel results: {len(all_results)}")
        if len(all_results) == 2:
            print("Two channels detected.")
            print("Channel 1 result:", all_results[0])
            print("Channel 2 result:", all_results[1])
        elif len(all_results) == 1:
            print("One channel detected.")
            print("Channel 1 result:", all_results[0])
        else:
            print("No channel data received.")

        return all_results