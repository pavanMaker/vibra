import numpy as np
import pandas as pd
from daqhats import mcc172, OptionFlags, SourceType
from scipy.fft import fft, fftfreq,ifft,rfft,rfftfreq,irfft
from scipy.signal import detrend, windows, sosfiltfilt, butter,find_peaks
from scipy.signal import firwin, filtfilt
# from scipy.integrate import cumulative_trapezoid
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import cmsisdsp as dsp
import json,os



class Mcc172Backend:
    def __init__(self, board_num=0, sample_rate=5000,  channel=[0, 1],buffer_size = 8192 ):
        self.board_num = board_num
        self.sample_rate = sample_rate
        print("Sample rate:", self.sample_rate)
       
        self.channel = channel
        self.board = mcc172(board_num)
        self.buffer_size = buffer_size
        print("Buffer size:", self.buffer_size)
        folder_name = 'pages'
        file_name = 'fir' \
        '_filters.json'
        filepath = os.path.join(folder_name,file_name)
        with open(filepath, 'r') as f:
            self.fir_filters = json.load(f)["filters"]

        self.firf32 = dsp.arm_fir_instance_f32()
        self.fir_state = None
        self.fir_coeffs = None
        self.current_fmax = None
        self.channel_data = {
            0: {
                'acc_peaks': [],
                'acc_rms': [],
                'acc_ptps': [],
                'vel_peaks': [],
                'vel_rms': [],
                'vel_ptps': [],
                'disp_peaks': [],
                'disp_rms': [],
                'disp_ptps': []
            },
            1: {
                'acc_peaks': [],
                'acc_rms': [],
                'acc_ptps': [],
                'vel_peaks': [],
                'vel_rms': [],
                'vel_ptps': [],
                'disp_peaks': [],
                'disp_rms': [],
                'disp_ptps': []
            }
        }
        
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
        result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=-1)
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

        print("No valid data received.")
        return np.zeros(0), np.zeros(0)

    def find_top5_peaks(self,x):
        peak_index = [-1] * 5
        peak_value = [-3.4e38] * 5

        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1]:
                for p in range(5):
                    if x[i] > peak_value[p]:
                        for k in range(4, p, -1):
                            peak_value[k] = peak_value[k - 1]
                            peak_index[k] = peak_index[k - 1]
                        peak_value[p] = x[i]
                        peak_index[p] = i
                        break

        return peak_index, peak_value


    def load_fir_from_json(self, fmax_hz):
        # Avoid reloading FIR if fmax did not change
        if self.current_fmax == fmax_hz:
            return

        for filt in self.fir_filters:
            
            if filt["fmax"] == fmax_hz:
                self.fir_coeffs = np.array(filt["coefficients"], dtype=np.float32)

                state_len = len(self.fir_coeffs) + self.buffer_size - 1
                self.fir_state = np.zeros(state_len, dtype=np.float32)

                dsp.arm_fir_init_f32(
                    self.firf32,
                    len(self.fir_coeffs),
                    self.fir_coeffs,
                    self.fir_state
                )

                self.current_fmax = fmax_hz

                print(f"✅ FIR loaded → fmax={fmax_hz} Hz | taps={len(self.fir_coeffs)}")
                return

        raise ValueError(f"No FIR found for fmax={fmax_hz}")


    def analyze(self, result_data, sensitivity, fmax_hz, fmin_hz, channel_id):
    
        calibiration_fac = 1.0
        g_to_m_s2 = 9.80665
        sampling_frequency = self.actual_rate
       
        self.load_fir_from_json(fmax_hz)


        # 1️⃣ Convert to acceleration + DC removal
        acceleration_g = result_data / (sensitivity * calibiration_fac)
        bin_size = sampling_frequency / len(acceleration_g)
       
        dc_offset = np.mean(acceleration_g)
        acceleration_waveform = acceleration_g - dc_offset
        

        # 2️⃣ Apply Fixed LPF FIR (0-1150 Hz passband)
        accel_lpf = dsp.arm_fir_f32(
            self.firf32,
            acceleration_waveform.astype(np.float32)
        )
        accel_lpf = accel_lpf[:len(acceleration_waveform)]

      
        N = len(acceleration_waveform)
        print("length of acceleration_waveform:",N)
        block_size = 1 / self.actual_rate
        frequencies_temp = rfftfreq(N, block_size)
      

       

        acc_fft_temp = rfft(acceleration_waveform)

        # 4️⃣ Soft HPF (remove frequencies < fmin)
        atten = 100000.0
        acc_fft_hpf = acc_fft_temp.copy()
        low_mask = frequencies_temp < fmin_hz
        acc_fft_hpf[low_mask] /= atten

        # 5️⃣ IFFT back to time domain
        accel_after_hpf = irfft(acc_fft_hpf)
        acceleration_waveform_final = accel_after_hpf

        # 6️⃣ Apply Hanning window
        window = windows.hann(N)
        accel_windowed = accel_after_hpf * window

        # Two different correction factors
        windowing_correction_amplitude = 2.0 / np.sum(window)
        U = np.sum(window**2) / N

        # 7️⃣ Second FFT (final spectrum)
        frequencies = rfftfreq(N, block_size)
        acc_fft = rfft(accel_windowed)

        # 8️⃣ Soft LPF (remove frequencies > fmax)
        acc_fft_filtered = acc_fft.copy()
        high_mask = frequencies > fmax_hz
        acc_fft_filtered[high_mask] /= atten

        # Final acceleration spectrum magnitude
        fft_mags = np.abs(acc_fft_filtered) * windowing_correction_amplitude

        # 9️⃣ Velocity integration
        omega = 2 * np.pi * frequencies
        valid = omega > 0

        vel_fft = np.zeros_like(acc_fft_filtered, dtype=complex)
        vel_fft[valid] = acc_fft_filtered[valid] / (1j * omega[valid])

        vel_fft_mags = np.abs(vel_fft) * windowing_correction_amplitude * g_to_m_s2 * 1000
        vel_time_waveform = irfft(vel_fft) * g_to_m_s2 * 1000

        # 🔟 Displacement integration
        disp_fft = np.zeros_like(acc_fft_filtered, dtype=complex)
        disp_fft[valid] = vel_fft[valid] / (1j * omega[valid])

        disp_fft_mags = np.abs(disp_fft) * windowing_correction_amplitude * g_to_m_s2 * 1_000_000
        disp_time_waveform = irfft(disp_fft) * g_to_m_s2 * 1_000_000

        # 1️⃣1️⃣ Acceleration metrics
        accel_abs = np.abs(acceleration_waveform_final)
        idx, peaks = self.find_top5_peaks(accel_abs)
        valid_peaks = [p for p in peaks if p > 0]
        overall_acc_peak = np.mean(valid_peaks) if valid_peaks else 0.0
        print("overall acceleration peak", overall_acc_peak)
        overall_acc_ptp = 2 * overall_acc_peak

        power_acc = np.abs(acc_fft_filtered[0])**2 + np.abs(acc_fft_filtered[-1])**2 \
                    + 2 * np.sum(np.abs(acc_fft_filtered[1:-1])**2)
        overall_acc_rms = np.sqrt(power_acc / U) / N

        # 1️⃣2️⃣ Velocity metrics
        power_vel = np.abs(vel_fft[0])**2 + np.abs(vel_fft[-1])**2 \
                    + 2 * np.sum(np.abs(vel_fft[1:-1])**2)
        overall_vel_rms = np.sqrt(power_vel / U) / N * g_to_m_s2 * 1000
        print("overall velocity rms:",overall_vel_rms)
        overall_vel_peak = overall_vel_rms * np.sqrt(2)
        print("overall velocity peak:",overall_vel_peak)
        overall_vel_ptp = 2 * overall_vel_peak

        # 1️⃣3️⃣ Displacement metrics
        power_disp = np.abs(disp_fft[0])**2 + np.abs(disp_fft[-1])**2 \
                    + 2 * np.sum(np.abs(disp_fft[1:-1])**2)
        overall_disp_rms = np.sqrt(power_disp / U) / N * g_to_m_s2 * 1_000_000
        overall_disp_peak = overall_disp_rms * np.sqrt(2)
        overall_disp_ptp = 2 * overall_disp_peak
        
        # 9️ Averaging per channel
        ch_data = self.channel_data[channel_id]
        
        ch_data['acc_peaks'].append(overall_acc_peak)
        ch_data['acc_rms'].append(overall_acc_rms**2)
        ch_data['acc_ptps'].append(overall_acc_ptp)
        ch_data['vel_peaks'].append(overall_vel_peak)
        ch_data['vel_rms'].append(overall_vel_rms**2)
        ch_data['vel_ptps'].append(overall_vel_ptp)
        ch_data['disp_peaks'].append(overall_disp_peak)
        ch_data['disp_rms'].append(overall_disp_rms**2)
        ch_data['disp_ptps'].append(overall_disp_ptp)
        
        for key in ch_data:
            if len(ch_data[key]) > 5:
                ch_data[key] = ch_data[key][-5:]
        
        overall_acc_peak = np.mean(ch_data['acc_peaks'])
        overall_acc_rms = np.sqrt(np.mean(ch_data['acc_rms']))
        overall_acc_ptp = np.mean(ch_data['acc_ptps'])
        overall_vel_peak = np.mean(ch_data['vel_peaks'])
        overall_vel_rms = np.sqrt(np.mean(ch_data['vel_rms']))
        overall_vel_ptp = np.mean(ch_data['vel_ptps'])
        overall_disp_peak = np.mean(ch_data['disp_peaks'])
        overall_disp_rms = np.sqrt(np.mean(ch_data['disp_rms']))
        overall_disp_ptp = np.mean(ch_data['disp_ptps'])
        
        return {
            "acceleration": acceleration_waveform,
            "velocity": vel_time_waveform,
            "displacement": disp_time_waveform,
            "time": np.linspace(0, N * block_size, N, endpoint=False),
            "fft_mags": fft_mags,
            "fft_mags_vel": vel_fft_mags,
            "fft_mags_disp": disp_fft_mags,
            "frequencies": frequencies,
            "acc_peak": overall_acc_peak,
            "acceleration_rms": overall_acc_rms,
            "acceleration_ptps": overall_acc_ptp,
            "velocity_rms": overall_vel_rms,
            "velocity_peak": overall_vel_peak,
            "velocity_ptps": overall_vel_ptp,
            "displacement_peak": overall_disp_peak,
            "displacement_rms": overall_disp_rms,
            "displacement_ptps": overall_disp_ptp
        }


    def get_latest_waveform(self, fmax_hz, fmin_hz, sensitivities):
        ch0_voltage, ch1_voltage = self.read_data()

        print(f"\n Received sensitivities from GUI → CH0: {sensitivities[0]} V/g, CH1: {sensitivities[1]} V/g")


        print(f"\n Analyzing Channel 0")
        result_ch0 = self.analyze(ch0_voltage,sensitivities[0],fmax_hz=fmax_hz,fmin_hz=fmin_hz,channel_id =0) if ch0_voltage.size > 0 else self.empty_result()
    
        print(f"\n Analyzing Channel 1")
        result_ch1 = self.analyze(ch1_voltage,sensitivities[1],fmax_hz=fmax_hz,fmin_hz=fmin_hz,channel_id =1) if ch1_voltage.size > 0 else self.empty_result()

        return result_ch0, result_ch1

 

    def empty_result(self):
        return {
            "acceleration": np.array([]),
            "velocity": np.array([]),
            "displacement": np.array([]),
            "time": np.array([]),
            "acc_peak": 0.0,
            "acc_rms": 0.0,
            "velocity_rms": 0.0,
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