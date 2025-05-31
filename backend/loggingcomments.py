# # def analyze_channel(self, channel0, channel1):
#     #     # converting voltage to acceleration in g
#     #     acceleration_g0 = channel0 / self.sensitivity
#     #     acceleration_g1 = channel1 / self.sensitivity
#     #     # detrending the acceleration data
#     #     acceleration_g0 = detrend(acceleration_g0,type='linear')
#     #     acceleration_g1 = detrend(acceleration_g1,type='linear')
#     #     # Applying bandpass filter to the acceleration dta
#     #     No_of_samples = len(acceleration_g0)
#     #     Time_1_sample = 1 / self.actual_rate
#     #     time_index = pd.timedelta_range(start='0s', periods=No_of_samples, freq=pd.Timedelta(seconds=Time_1_sample))
#     #     acceleration_df0 = pd.DataFrame(acceleration_g0, index=time_index, columns=['acceleration'])
#     #     acceleration_df1 = pd.DataFrame(acceleration_g1, index=time_index, columns=['acceleration'])
#     #     # Applying bandpass filter to the acceleration dta
#     #     acceleration_df0 = butterworth(acceleration_df0, low_cutoff=3, high_cutoff=self.fmax, half_order=3)
#     #     acceleration_df1 = butterworth(acceleration_df1, low_cutoff=3, high_cutoff=self.fmax, half_order=3)
#     #     # convertion of acceleration data which is in dataframe to numpy array
#     #     acceleration_g0 = acceleration_df0['acceleration'].to_numpy()
#     #     acceleration_g1 = acceleration_df1['acceleration'].to_numpy()
#     #     #applying hanning for acceleration data
#     #     acceleration_window0 = windows.hann(len(acceleration_g0))
#     #     acceleration_window1 = windows.hann(len(acceleration_g1))
#     #     acceleration_windowed0 = acceleration_g0 * acceleration_window0
#     #     acceleration_windowed1 = acceleration_g1 * acceleration_window1
#     #     # window correction
#     #     window_correction0 = np.sqrt(np.mean(acceleration_window0 ** 2))
#     #     window_correction1 = np.sqrt(np.mean(acceleration_window1 ** 2))
#     #     # Convert to m/s²
#     #     acceleration_m_s2_0 = acceleration_g0 * 9.80665  # Convert to m/s²
#     #     acceleration_m_s2_1 = acceleration_g1 * 9.80665
#     #     # Integrate to velocity and displacement
#     #     df0 = pd.DataFrame(acceleration_m_s2_0, index=time_index, columns=["acceleration"])
#     #     df1 = pd.DataFrame(acceleration_m_s2_1, index=time_index, columns=["acceleration"])
#     #     integrated0 = integrals(df0, n=2, zero="mean", tukey_percent=0.05)
#     #     integrated1 = integrals(df1, n=2, zero="mean", tukey_percent=0.05)
#     #     velocity_m_s_0 = integrated0[1]["acceleration"].to_numpy()
#     #     velocity_m_s_1 = integrated1[1]["acceleration"].to_numpy()
#     #     displacement_m_0 = integrated0[2]["acceleration"].to_numpy()    
#     #     displacement_m_1 = integrated1[2]["acceleration"].to_numpy()
#     #     # Step 1: Detrend velocity  
#     #     velocity_m_s_0 = detrend(velocity_m_s_0, type='linear')
#     #     velocity_m_s_1 = detrend(velocity_m_s_1, type='linear')
#     #     # Step 2: Bandpass filter for velocity
#     #     # Applied band pass filter using Butterworth filter
#     #     velocity_df0 = pd.DataFrame(velocity_m_s_0, index=df0.index, columns=['velocity'])
#     #     velocity_df1 = pd.DataFrame(velocity_m_s_1, index=df1.index, columns=['velocity'])
#     #     velocity_df0 = butterworth(velocity_df0, low_cutoff=3, high_cutoff=500, half_order=3)
#     #     velocity_df1 = butterworth(velocity_df1, low_cutoff=3, high_cutoff=500, half_order=3)
#     #     velocity_m_s_0 = velocity_df0['velocity'].to_numpy()
#     #     velocity_m_s_1 = velocity_df1['velocity'].to_numpy()
#     #     # Step 3: Apply Hanning window before FFT for velocity
#     #     velocity_mm_s_0 = velocity_m_s_0 * 1000  # Convert to mm/s
#     #     velocity_mm_s_1 = velocity_m_s_1 * 1000
#     #     window_v0 = windows.hann(len(velocity_mm_s_0))
#     #     window_v1 = windows.hann(len(velocity_mm_s_1))
#     #     velocity_windowed0 = velocity_mm_s_0 * window_v0
#     #     velocity_windowed1 = velocity_mm_s_1 * window_v1
#     #     # window correction
#     #     window_correction_v0 = np.sqrt(np.mean(window_v0 ** 2))
#     #     window_correction_v1 = np.sqrt(np.mean(window_v1 ** 2))
#     #     # Step 4: Detrend displacement
#     #     displacement_m_0 = detrend(displacement_m_0, type='linear')
#     #     displacement_m_1 = detrend(displacement_m_1, type='linear')
        
#     #     displacement_df0 = pd.DataFrame(displacement_m_0, index=df0.index, columns=['displacement'])
#     #     displacement_df1 = pd.DataFrame(displacement_m_1, index=df1.index, columns=['displacement'])
#     #     # bandpass filter for displacement
#     #     displacement_df0 = butterworth(displacement_df0, low_cutoff=3, high_cutoff=500, half_order=3)
#     #     displacement_df1 = butterworth(displacement_df1, low_cutoff=3, high_cutoff=500, half_order=3)
#     #     displacement_m_s_0 = displacement_df0['displacement'].to_numpy()
#     #     displacement_m_s_1 = displacement_df1['displacement'].to_numpy()
#     #     # Convert to µm
#     #     displacement_um_0 = displacement_m_s_0 * 1e6  # Convert to µm
#     #     displacement_um_1 = displacement_m_s_1 * 1e6
#     #     # Apply Hanning window for displacement
#     #     window_d0 = windows.hann(len(displacement_um_0))
#     #     window_d1 = windows.hann(len(displacement_um_1))
#     #     displacement_windowed0 = displacement_um_0 * window_d0
#     #     displacement_windowed1 = displacement_um_1 * window_d1
#     #     # window correction
#     #     window_correction_d0 = np.sqrt(np.mean(window_d0 ** 2))
#     #     window_correction_d1 = np.sqrt(np.mean(window_d1 ** 2))
#     #     # fft for acceleration 
#     #     N0 = len(acceleration_windowed0)
#     #     N1 = len(acceleration_windowed1)
#     #     T = 1 / self.actual_rate
#     #     freqs0 = fftfreq(N0, T)
#     #     freqs1 = fftfreq(N1, T)
#     #     fft_result0 = fft(acceleration_windowed0)
#     #     fft_result1 = fft(acceleration_windowed1)
#     #     fft_mags0 = (2.0 / N0) * np.abs(fft_result0[:N0 // 2])
#     #     fft_mags1 = (2.0 / N1) * np.abs(fft_result1[:N1 // 2])
#     #     pos_freqs0 = freqs0[:N0 // 2]
#     #     pos_freqs1 = freqs1[:N1 // 2]
#     #     dom_freq0 = pos_freqs0[np.argmax(fft_mags0)]
#     #     dom_freq1 = pos_freqs1[np.argmax(fft_mags1)]
#     #     # fft for velocity
#     #     N2 = len(velocity_windowed0)
#     #     N3 = len(velocity_windowed1)
#     #     T1 = 1 / self.actual_rate
#     #     freqs_vel0 = fftfreq(N2, T1)
#     #     freqs_vel1 = fftfreq(N3, T1)
#     #     fft_result_vel0 = fft(velocity_windowed0)
#     #     fft_result_vel1 = fft(velocity_windowed1)
#     #     fft_mags_vel0 = (2.0 / N2) * np.abs(fft_result_vel0[:N2 // 2])
#     #     fft_mags_vel1 = (2.0 / N3) * np.abs(fft_result_vel1[:N3 // 2])
#     #     pos_freqs_vel0 = freqs_vel0[:N2 // 2]
#     #     pos_freqs_vel1 = freqs_vel1[:N3 // 2]
#     #     # fft for displacement
#     #     N4 = len(displacement_windowed0)
#     #     N5 = len(displacement_windowed1)
#     #     T2 = 1 / self.actual_rate
#     #     freqs_disp0 = fftfreq(N4, T2)
#     #     freqs_disp1 = fftfreq(N5, T2)
#     #     fft_result_disp0 = fft(displacement_windowed0)
#     #     fft_result_disp1 = fft(displacement_windowed1)
#     #     fft_mags_disp0 = (2.0 / N4) * np.abs(fft_result_disp0[:N4 // 2])
#     #     fft_mags_disp1 = (2.0 / N5) * np.abs(fft_result_disp1[:N5 // 2])
#     #     pos_freqs_disp0 = freqs_disp0[:N4 // 2]
#     #     pos_freqs_disp1 = freqs_disp1[:N5 // 2]
#     #     # acceleration peak value
#     #     acc_peak0 = np.max(np.abs(acceleration_windowed0))
#     #     acc_peak1 = np.max(np.abs(acceleration_windowed1))
#     #     # velocity rms value by wveform and rms in fft
#     #     vel_rms0 = rms(pd.Series(velocity_windowed0))
#     #     vel_rms1 = rms(pd.Series(velocity_windowed1))
#     #     rms_fft0 = np.sqrt(np.sum((fft_mags_vel0[1:] ** 2) / 2)) / window_correction_v0
#     #     rms_fft1 = np.sqrt(np.sum((fft_mags_vel1[1:] ** 2) / 2)) / window_correction_v1
#     #     # displacement peak to peak value
#     #     disp_pp0 = np.ptp(displacement_windowed0)
#     #     disp_pp1 = np.ptp(displacement_windowed1)

#     #     return {
#     #         "acceleration0": acceleration_g0,
#     #         "acceleration1": acceleration_g1,
#     #         "velocity0": velocity_mm_s_0,
#     #         "velocity1": velocity_mm_s_1,
#     #         "displacement0": displacement_um_0,
#     #         "displacement1": displacement_um_1,
#     #         "time": np.linspace(0, len(acceleration_g0) * (1 / self.actual_rate), len(acceleration_g0), endpoint=False),
#     #         "acc_peak0": acc_peak0,
#     #         "acc_peak1": acc_peak1,
#     #         "vel_rms0": vel_rms0,
#     #         "vel_rms1": vel_rms1,
#     #         "disp_pp0": disp_pp0,
#     #         "disp_pp1": disp_pp1,
#     #         "dom_freq0": dom_freq0,
#     #         "dom_freq1": dom_freq1,
#     #         "fft_freqs_accel0": pos_freqs0,
#     #         "fft_mags_accel0": fft_mags0,
#     #         "fft_freqs_accel1": pos_freqs1,
#     #         "fft_mags_accel1": fft_mags1,
#     #         "fft_freqs_velo0": pos_freqs_vel0,
#     #         "fft_mags_velo0": fft_mags_vel0,
#     #         "fft_freqs_velo1": pos_freqs_vel1,
#     #         "fft_mags_velo1": fft_mags_vel1,
#     #         "fft_freqs_disp0": pos_freqs_disp0,
#     #         "fft_mags_disp0": fft_mags_disp0,
#     #         "fft_freqs_disp1": pos_freqs_disp1,
#     #         "fft_mags_disp1": fft_mags_disp1
#     #     }
#     # def get_latest_waveform_channel(self):
#     #     channel0, channel1 = self.read_data()
#     #     if len(channel0) == 0 or len(channel1) == 0:
#     #         print("No data received from MCC 172.")
#     #         return [], [], [], [], 0, 0, 0, 0, 0
#     #     result = self.analyze_channel(channel0, channel1)
#     #     return (
#     #         result["time"].tolist(),
#     #         result["acceleration0"].tolist(),
#     #         result["acceleration1"].tolist(),
#     #         result["velocity0"].tolist(),
#     #         result["velocity1"].tolist(),
#     #         result["displacement0"].tolist(),
#     #         result["displacement1"].tolist(),
#     #         result["acc_peak0"],
#     #         result["acc_peak1"],
#     #         result["vel_rms0"],
#     #         result["vel_rms1"],
#     #         result["disp_pp0"],
#     #         result["disp_pp1"],
#     #         result["dom_freq0"],
#     #         result["dom_freq1"],
#     #         result["fft_freqs_accel0"],
#     #         result["fft_mags_accel0"],
#     #         result["fft_freqs_accel1"],
#     #         result["fft_mags_accel1"],
#     #         result["fft_freqs_velo0"],
#     #         result["fft_mags_velo0"],
#     #         result["fft_freqs_velo1"],
#     #         result["fft_mags_velo1"],
#     #         result["fft_freqs_disp0"],
#     #         result["fft_mags_disp0"],
#     #         result["fft_freqs_disp1"],
#     #         result["fft_mags_disp1"]
#     #     )
        
        
#     # def analyze(self, result_data):
#     #     # Convert voltage to acceleration in g
#     #     acceleration_g = result_data / self.sensitivity
#     #     acceleration_g = detrend(acceleration_g, type='linear')  # Remove trend
#     #     #Apply bandpass filter
#     #     #acceleration_df = pd.DataFrame(acceleration_g, columns=['acceleration'])
#     #     print("high_cutoff",self.fmax)
       
                                             
#     #     #acceleration_df = butterworth(acceleration_df,low_cutoff=3, high_cutoff=self.fmax, half_order=3)
#     #     #acceleration_df = butterworth(acceleration_df, low_cutoff=3, high_cutoff=500, half_order=3)
#     #     #acceleration_g = acceleration_df['acceleration'].to_numpy()
#     #     #acceleration_m_s2 = acceleration_g * 9.80665  # Convert to m/s²

#     #     # Time setup
#     #     N = len(acceleration_g)
#     #     T = 1 / self.actual_rate
#     #     time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
#     #     #df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])
#     #     #Applying band pass filter to the acceleration for waveform
#     #     df = pd.DataFrame(acceleration_g, index=time_index, columns=['acceleration'])
#     #     acceleration_g = butterworth(df, low_cutoff=3, high_cutoff=self.fmax, half_order=3)
#     #     acceleration_g = df['acceleration'].to_numpy()
#     #     # Convert to m/s² to integrate into velocity and displacement
#     #     acceleration_m_s2 = acceleration_g * 9.80665  # Convert to m/s²

#     #     # Integrate to velocity and displacement
#     #     df1 = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])
#     #     integrated = integrals(df1, n=2, zero="mean", tukey_percent=0.05)
#     #     velocity_m_s = integrated[1]["acceleration"].to_numpy()
#     #     displacement_m = integrated[2]["acceleration"].to_numpy()

#     #     # Step 1: Detrend velocity
#     #     velocity_m_s = detrend(velocity_m_s, type='linear')

#     #     # Step 2: Bandpass filter for velocity
#     #     # Applied band pass filter using Butterworth filter
#     #     velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
#     #     velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=500, half_order=3)
#     #     velocity_m_s = velocity_df['velocity'].to_numpy()

#     #     # Step 3: Apply Hanning window before FFT
#     #     velocity_mm_s = velocity_m_s * 1000  # Convert to mm/s
#     #     window = windows.hann(len(velocity_mm_s))
#     #     velocity_windowed = velocity_mm_s * window
#     #     window_correction = np.sqrt(np.mean(window ** 2))

#     #     # step 4: Detrend displacement  
#     #     displacement_m = detrend(displacement_m, type='linear')

#     #     # Step 5: Bandpass filter for displacement
#     #     displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
#     #     # Use fmax from AnalysisParameters if provided, otherwise default to 500
       
#     #     displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=500,half_order=3)
#     #     displacement_m_s = displacement_df['displacement'].to_numpy()

#     #     displacement_um = displacement_m_s * 1e6  # Convert to µm
#     #     window_d = windows.hann(len(displacement_um))
#     #     displacement_windowed = displacement_um * window_d
#     #     window_correction_d = np.sqrt(np.mean(window_d ** 2))

#     #     N1 = len(velocity_mm_s)
#     #     N2 = len(displacement_um)
       

#     #     # RMS and metrics
#     #     vel_rms = rms(pd.Series(velocity_mm_s))
#     #     disp_pp = np.ptp(displacement_um)
#     #     acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
#     #     acc_peak = np.max(np.abs(acceleration_g))

#     #     # FFT for acceleration
#     #     fft_result = fft(acceleration_g)
#     #     freqs = fftfreq(N, T)
#     #     fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
#     #     pos_freqs = freqs[:N // 2]
#     #     dom_freq = pos_freqs[np.argmax(fft_mags)]

#     #     # FFT for velocity (with windowing)
#     #     fft_result_vel = fft(velocity_windowed)
#     #     freqs_vel = fftfreq(N1, T)
#     #     fft_mags_vel = (2.0 / N1) * np.abs(fft_result_vel[:N1 // 2])
#     #     pos_freqs_vel = freqs_vel[:N1 // 2]
#     #     #dom_freq_vel = pos_freqs_vel[np.argmax(fft_mags_vel)]

#     #     #FFT for displacement (with windowing)
#     #     fft_result_disp = fft(displacement_windowed)
#     #     freqs_disp = fftfreq(N2, T)
#     #     fft_mags_disp = (2.0 / N2) * np.abs(fft_result_disp[:N2 // 2])
#     #     pos_freqs_disp = freqs_disp[:N2 // 2]
#     #     #dom_freq_disp = pos_freqs_disp[np.argmax(fft_mags_disp)]


#     #     # Normalize RMS FFT for windowing loss
#     #     rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / window_correction
#     #     #velocity peak vaalue
#     #     vel_peak = np.max(np.abs(velocity_mm_s))
#     #     print("Velocity Peak (mm/s):", vel_peak)
#     #     vel_peak1= np.max(np.abs(velocity_windowed))
#     #     print("Velocity Peak (mm/s) with windowing:", vel_peak1)
#     #     # displacement peak value
#     #     disp_peak = np.max(np.abs(displacement_um))
#     #     disp_peak1 = np.max(np.abs(displacement_windowed))
#     #     print("Displacement Peak (um):", disp_peak)
#     #     print("Displacement Peak (um) with windowing:", disp_peak1)
#     #     print("Displacement Peak (um):", disp_pp)

#     #     print("Velocity RMS (mm/s):", vel_rms)
#     #     print("Dominant Frequency (Hz):", dom_freq)
#     #     print("RMS FFT:", rms_fft)

#     #     return {
#     #         "acceleration": acceleration_g,
#     #         "velocity": velocity_mm_s,
#     #         "displacement": displacement_um,
#     #         "time": np.linspace(0, N * T, N, endpoint=False),
#     #         "acc_peak": acc_peak,
#     #         "acc_rms": acc_rms,
#     #         "vel_rms": vel_rms,
#     #         "disp_pp": disp_pp,
#     #         "dom_freq": dom_freq,
#     #         "fft_freqs": pos_freqs,
#     #         "fft_mags": fft_mags,
#     #         "freqs_vel": pos_freqs_vel,
#     #         "fft_mags_vel": fft_mags_vel,
#     #         "fft_freqs_disp": pos_freqs_disp,
#     #         "fft_mags_disp": fft_mags_disp,
#     #         #"dom_freq_vel": dom_freq_vel,
#     #         "rms_fft": rms_fft
#     #     }

#     # def get_latest_waveform(self):
#     #     result_data = self.read_data()
#     #     if len(result_data) == 0:
#     #         print("No data received from MCC 172.")
#     #         return [], [], [], [], 0, 0, 0, 0, 0

#     #     result = self.analyze(result_data)
#     #     return (
#     #         result["time"].tolist(),
#     #         result["acceleration"].tolist(),
#     #         result["velocity"].tolist(),
#     #         result["displacement"].tolist(),
#     #         result["acc_peak"],
#     #         result["acc_rms"],
#     #         result["vel_rms"],
#     #         result["disp_pp"],
#     #         result["dom_freq"],
#     #         result["fft_freqs"],
#     #         result["fft_mags"],
#     #         result["freqs_vel"],
#     #         result["fft_mags_vel"],
#     #         result["fft_freqs_disp"],
#     #         result["fft_mags_disp"],
#     #         #result["dom_freq_vel"],
#     #         result["rms_fft"]
#     #     )







#  # def update_plots(self):
#     # res0, res1 = self.daq.get_latest_waveform_channel()
#     # if res0 is None:
#     #     return  # nothing yet

#     # # pick which measurement set to show
#     # if self.selected_quantity == "Velocity":
#     #     y0, y1 = res0["vel"],  res1["vel"]
#     #     f0, f1 = res0["fft_vel_m"], res1["fft_vel_m"]
#     #     fft_freqs = res0["fft_acc_f"]     # same freqs for all
#     #     y_label = "Velocity (mm/s)"
#     # elif self.selected_quantity == "Displacement":
#     #     y0, y1 = res0["disp"], res1["disp"]
#     #     f0, f1 = res0["fft_disp_m"], res1["fft_disp_m"]
#     #     fft_freqs = res0["fft_acc_f"]
#     #     y_label = "Displacement (µm)"
#     # else:  # Acceleration
#     #     y0, y1 = res0["acc"], res1["acc"]
#     #     f0, f1 = res0["fft_acc_m"], res1["fft_acc_m"]
#     #     fft_freqs = res0["fft_acc_f"]
#     #     y_label = "Acceleration (g)"

#     # # --- now plot based on current trace mode ---------------------
#     # mode = self.stacked_views.currentIndex()

#     # if mode == 0:       # Readings + Waveform (Ch-0 only)
#     #     self._draw_wave(self.ax_waveform, self.canvas_waveform,
#     #                     res0["time"], y0, y_label)

#     # elif mode == 1:     # Waveform + Spectrum (Ch-0 only)
#     #     self._draw_wave(self.ax_top,    self.canvas_top,
#     #                     res0["time"], y0, y_label)
#     #     self._draw_fft (self.ax_bottom, self.canvas_bottom,
#     #                     fft_freqs, f0, y_label)

#     # elif mode == 2:     # Ch-0 + Ch-1 waveform
#     #     self._draw_wave(self.ax_top,    self.canvas_top,
#     #                     res0["time"], y0, y_label, title="Ch-0")
#     #     self._draw_wave(self.ax_bottom, self.canvas_bottom,
#     #                     res1["time"], y1, y_label, title="Ch-1")

#     # elif mode == 3:     # Ch-0 + Ch-1 spectrum
#     #     self._draw_fft(self.ax_top,    self.canvas_top,
#     #                    fft_freqs, f0, y_label, title="Ch-0 FFT")
#     #     self._draw_fft(self.ax_bottom, self.canvas_bottom,
#     #                    fft_freqs, f1, y_label, title="Ch-1 FFT")

#     # # update numeric read-outs from Channel-0 (as the CoCo-80X does)
#     # self.acc_input["input"].setText(f"{res0['acc_peak']:.2f}")
#     # self.vel_input["input"].setText(f"{res0['vel_rms']:.2f}")
#     # self.disp_input["input"].setText(f"{res0['disp_pp']:.2f}")
#     # self.freq_input["input"].setText(f"{res0['dom_freq']:.2f}")

         

#     # def update_plots(self):
#     #     # Unpack the result from get_latest_waveform for dual channel
#     #     (
#     #         t, accel0, accel1, velocity0, velocity1, displacement0, displacement1,
#     #         acc_peak0, acc_peak1, vel_rms0, vel_rms1, disp_pp0, disp_pp1,
#     #         dom_freq0, dom_freq1,
#     #         fft_freqs_accel0, fft_mags_accel0, fft_freqs_accel1, fft_mags_accel1,
#     #         fft_freqs_velo0, fft_mags_velo0, fft_freqs_velo1, fft_mags_velo1,
#     #         fft_freqs_disp0, fft_mags_disp0, fft_freqs_disp1, fft_mags_disp1
#     #     ) = self.daq.get_latest_waveform_channel()

#     #     if self.selected_quantity == "Velocity":
#     #         y_data = velocity0
#     #         y_label = "Velocity (mm/s)"
#     #         pos_freqs = fft_freqs_velo0[:len(y_data) // 2]
#     #         mask = pos_freqs <= self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_velo0[mask]

#     #         y_data1 = velocity1
#     #         y_label1 = "Velocity (mm/s)"
#     #         pos_freqs1 = fft_freqs_velo1[:len(y_data1) // 2]
#     #         mask1 = pos_freqs1 <= self.daq.fmax 
#     #         self.freqs1 = pos_freqs1[mask1]
#     #         fft_mags1 = fft_mags_velo1[mask1]
#     #     elif self.selected_quantity == "Displacement":
#     #         y_data = displacement0
#     #         y_label = "Displacement (μm)"
#     #         pos_freqs = fft_freqs_disp0[:len(y_data) // 2]
#     #         mask = pos_freqs <= self.daq.fmax 
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_disp0[mask]

#     #         y_data1 = displacement1
#     #         y_label1 = "Displacement (μm)"
#     #         pos_freqs1 = fft_freqs_disp1[:len(y_data1) // 2]
#     #         mask1 = pos_freqs1 <= self.daq.fmax 
#     #         self.freqs1 = pos_freqs1[mask1]
#     #         fft_mags1 = fft_mags_disp1[mask1]
#     #     else:
#     #         y_data = accel0
#     #         y_label = "Acceleration (g)"
#     #         pos_freqs = fft_freqs_accel0[:len(y_data) // 2]
#     #         mask = pos_freqs <= self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_accel0[mask]

#     #         y_data1 = accel1
#     #         y_label1 = "Acceleration (g)"
#     #         pos_freqs1 = fft_freqs_accel1[:len(y_data1) // 2]
#     #         mask1 = pos_freqs1 <= self.daq.fmax
#     #         self.freqs1 = pos_freqs1[mask1]
#     #         fft_mags1 = fft_mags_accel1[mask1]
#     #     # Update the waveform plot
#     #     view_index = self.stacked_views.currentIndex()
#     #     if view_index == 0:
#     #         self.ax_waveform.clear()
#     #         self.ax_waveform.plot(t, y_data)
#     #         self.ax_waveform.set_title("Waveform")
#     #         self.ax_waveform.set_xlabel("Time (s)")
#     #         self.ax_waveform.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_waveform.grid(True)
#     #         self.canvas_waveform.draw()
#     #     elif view_index == 1:
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(t, y_data)
#     #         self.ax_top.set_title("Waveform")
#     #         self.ax_top.set_xlabel("Time (s)")
#     #         self.ax_top.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         # Update the spectrum plot
#     #         self.ax_bottom.clear()
#     #         self.ax_bottom.plot(self.freqs, fft_mags)
#     #         self.ax_bottom.set_title("Spectrum")
#     #         self.ax_bottom.set_xlabel("Frequency (Hz)")
#     #         self.ax_bottom.set_ylabel(f"{y_label} RMS")
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()
#     #     elif view_index == 2:
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(t, y_data)
#     #         self.ax_top.set_title("Waveform (Channel 0)")
#     #         self.ax_top.set_xlabel("Time (s)")
#     #         self.ax_top.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         # Update the waveform plot for channel 1
#     #         self.ax_bottom.clear()
#     #         self.ax_bottom.plot(t, y_data1)
#     #         self.ax_bottom.set_title("Waveform (Channel 1)")
#     #         self.ax_bottom.set_xlabel("Time (s)")
#     #         self.ax_bottom.set_ylabel(y_label1)
#     #         margin1 = (max(y_data1) - min(y_data1)) * 0.1 or 0.2
#     #         self.ax_bottom.set_ylim(min(y_data1) - margin1, max(y_data1) + margin1)
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()

#     #     elif view_index == 3:
#     #         # Update the spectrum plot for channel 1
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(self.freqs, fft_mags)
#     #         self.ax_top.set_title("Spectrum (Channel 0)")
#     #         self.ax_top.set_xlabel("Frequency (Hz)")
#     #         self.ax_top.set_ylabel(f"{y_label} RMS")
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         # Update the spectrum plot for channel 2
#     #         self.ax_bottom.clear()
#     #         self.ax_bottom.plot(self.freqs1, fft_mags1)
#     #         self.ax_bottom.set_title("Spectrum (Channel 1)")
#     #         self.ax_bottom.set_xlabel("Frequency (Hz)")
#     #         self.ax_bottom.set_ylabel(f"{y_label1} RMS")
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()
#     #     # Update the readings
#     #     self.acc_input["input"].setText(f"{acc_peak0:.2f}")
#     #     self.vel_input["input"].setText(f"{vel_rms0:.2f}")
#     #     self.disp_input["input"].setText(f"{disp_pp0:.2f}")
#     #     self.freq_input["input"].setText(f"{dom_freq0:.2f}")
            


#     # def update_plot(self):
#     #     self.t, self.accel, self.velocity, self.displacement, acc_peak, acc_rms, vel_rms, disp_pp, dom_freq,fft_freqs,fft_mags,freqs_vel,fft_mags_vel,fft_freqs_disp,fft_mags_disp,rms_fft= self.daq.get_latest_waveform()
#     #     if len(self.t) == 0:
#     #         return


#     #     if self.selected_quantity == "Velocity":
#     #         y_data = self.velocity
#     #         N = len(y_data)
#     #         y_label = "Velocity (mm/s)"
#     #         pos_freqs = freqs_vel[:N // 2]
#     #         mask = pos_freqs <= self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_vel[mask]
#     #         self.fft_mags = fft_mags


#     #     elif self.selected_quantity == "Displacement":
#     #         y_data = self.displacement
#     #         y_label = "Displacement (μm)"
#     #         N2 = len(y_data)
#     #         pos_freqs = fft_freqs_disp[:N2 // 2]
#     #         mask = pos_freqs <= self.daq.fmax 
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_disp[mask]
#     #         self.fft_mags = fft_mags
            
#     #     else:
#     #         y_data = self.accel
#     #         N1 = len(y_data)
#     #         y_label = "Acceleration (g)"
#     #         pos_freqs = fft_freqs[:N1 // 2]
#     #         mask = pos_freqs <=self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags[mask]
#     #         self.fft_mags = fft_mags

#     #     view_index = self.stacked_views.currentIndex()
#     #     if view_index == 0:
#     #         self.ax_waveform.clear()
#     #         self.ax_waveform.plot(self.t, y_data)
#     #         self.ax_waveform.set_title("Waveform")
#     #         self.ax_waveform.set_xlabel("Time (s)")
#     #         self.ax_waveform.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_waveform.grid(True)
#     #         self.canvas_waveform.draw()
#     #     else:
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(self.t, y_data)
#     #         self.ax_top.set_title("Waveform")
#     #         self.ax_top.set_xlabel("Time (s)")
#     #         self.ax_top.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         self.ax_bottom.clear()
#     #         #fft_result = np.fft.fft(y_data)
            
#     #         #freqs = np.fft.fftfreq(N, 1 / self.daq.actual_rate)
#     #         #fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
            
#     #         self.ax_bottom.plot(self.freqs, self.fft_mags)
#     #         self.ax_bottom.set_title("Spectrum")
#     #         self.ax_bottom.set_xlabel("Frequency (Hz)")
#     #         self.ax_bottom.set_ylabel(f"{y_label} RMS")
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()

#     #     self.acc_input["input"].setText(f"{acc_peak:.2f}")
#     #     self.vel_input["input"].setText(f"{rms_fft:.2f}")
#     #     self.disp_input["input"].setText(f"{disp_pp:.2f}")
#     #     self.freq_input["input"].setText(f"{dom_freq:.2f}")


#     # def update_plots(self):
#     # res0, res1 = self.daq.get_latest_waveform_channel()
#     # if res0 is None:
#     #     return  # nothing yet

#     # # pick which measurement set to show
#     # if self.selected_quantity == "Velocity":
#     #     y0, y1 = res0["vel"],  res1["vel"]
#     #     f0, f1 = res0["fft_vel_m"], res1["fft_vel_m"]
#     #     fft_freqs = res0["fft_acc_f"]     # same freqs for all
#     #     y_label = "Velocity (mm/s)"
#     # elif self.selected_quantity == "Displacement":
#     #     y0, y1 = res0["disp"], res1["disp"]
#     #     f0, f1 = res0["fft_disp_m"], res1["fft_disp_m"]
#     #     fft_freqs = res0["fft_acc_f"]
#     #     y_label = "Displacement (µm)"
#     # else:  # Acceleration
#     #     y0, y1 = res0["acc"], res1["acc"]
#     #     f0, f1 = res0["fft_acc_m"], res1["fft_acc_m"]
#     #     fft_freqs = res0["fft_acc_f"]
#     #     y_label = "Acceleration (g)"

#     # # --- now plot based on current trace mode ---------------------
#     # mode = self.stacked_views.currentIndex()

#     # if mode == 0:       # Readings + Waveform (Ch-0 only)
#     #     self._draw_wave(self.ax_waveform, self.canvas_waveform,
#     #                     res0["time"], y0, y_label)

#     # elif mode == 1:     # Waveform + Spectrum (Ch-0 only)
#     #     self._draw_wave(self.ax_top,    self.canvas_top,
#     #                     res0["time"], y0, y_label)
#     #     self._draw_fft (self.ax_bottom, self.canvas_bottom,
#     #                     fft_freqs, f0, y_label)

#     # elif mode == 2:     # Ch-0 + Ch-1 waveform
#     #     self._draw_wave(self.ax_top,    self.canvas_top,
#     #                     res0["time"], y0, y_label, title="Ch-0")
#     #     self._draw_wave(self.ax_bottom, self.canvas_bottom,
#     #                     res1["time"], y1, y_label, title="Ch-1")

#     # elif mode == 3:     # Ch-0 + Ch-1 spectrum
#     #     self._draw_fft(self.ax_top,    self.canvas_top,
#     #                    fft_freqs, f0, y_label, title="Ch-0 FFT")
#     #     self._draw_fft(self.ax_bottom, self.canvas_bottom,
#     #                    fft_freqs, f1, y_label, title="Ch-1 FFT")

#     # # update numeric read-outs from Channel-0 (as the CoCo-80X does)
#     # self.acc_input["input"].setText(f"{res0['acc_peak']:.2f}")
#     # self.vel_input["input"].setText(f"{res0['vel_rms']:.2f}")
#     # self.disp_input["input"].setText(f"{res0['disp_pp']:.2f}")
#     # self.freq_input["input"].setText(f"{res0['dom_freq']:.2f}")

         

#     # def update_plots(self):
#     #     # Unpack the result from get_latest_waveform for dual channel
#     #     (
#     #         t, accel0, accel1, velocity0, velocity1, displacement0, displacement1,
#     #         acc_peak0, acc_peak1, vel_rms0, vel_rms1, disp_pp0, disp_pp1,
#     #         dom_freq0, dom_freq1,
#     #         fft_freqs_accel0, fft_mags_accel0, fft_freqs_accel1, fft_mags_accel1,
#     #         fft_freqs_velo0, fft_mags_velo0, fft_freqs_velo1, fft_mags_velo1,
#     #         fft_freqs_disp0, fft_mags_disp0, fft_freqs_disp1, fft_mags_disp1
#     #     ) = self.daq.get_latest_waveform_channel()

#     #     if self.selected_quantity == "Velocity":
#     #         y_data = velocity0
#     #         y_label = "Velocity (mm/s)"
#     #         pos_freqs = fft_freqs_velo0[:len(y_data) // 2]
#     #         mask = pos_freqs <= self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_velo0[mask]

#     #         y_data1 = velocity1
#     #         y_label1 = "Velocity (mm/s)"
#     #         pos_freqs1 = fft_freqs_velo1[:len(y_data1) // 2]
#     #         mask1 = pos_freqs1 <= self.daq.fmax 
#     #         self.freqs1 = pos_freqs1[mask1]
#     #         fft_mags1 = fft_mags_velo1[mask1]
#     #     elif self.selected_quantity == "Displacement":
#     #         y_data = displacement0
#     #         y_label = "Displacement (μm)"
#     #         pos_freqs = fft_freqs_disp0[:len(y_data) // 2]
#     #         mask = pos_freqs <= self.daq.fmax 
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_disp0[mask]

#     #         y_data1 = displacement1
#     #         y_label1 = "Displacement (μm)"
#     #         pos_freqs1 = fft_freqs_disp1[:len(y_data1) // 2]
#     #         mask1 = pos_freqs1 <= self.daq.fmax 
#     #         self.freqs1 = pos_freqs1[mask1]
#     #         fft_mags1 = fft_mags_disp1[mask1]
#     #     else:
#     #         y_data = accel0
#     #         y_label = "Acceleration (g)"
#     #         pos_freqs = fft_freqs_accel0[:len(y_data) // 2]
#     #         mask = pos_freqs <= self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_accel0[mask]

#     #         y_data1 = accel1
#     #         y_label1 = "Acceleration (g)"
#     #         pos_freqs1 = fft_freqs_accel1[:len(y_data1) // 2]
#     #         mask1 = pos_freqs1 <= self.daq.fmax
#     #         self.freqs1 = pos_freqs1[mask1]
#     #         fft_mags1 = fft_mags_accel1[mask1]
#     #     # Update the waveform plot
#     #     view_index = self.stacked_views.currentIndex()
#     #     if view_index == 0:
#     #         self.ax_waveform.clear()
#     #         self.ax_waveform.plot(t, y_data)
#     #         self.ax_waveform.set_title("Waveform")
#     #         self.ax_waveform.set_xlabel("Time (s)")
#     #         self.ax_waveform.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_waveform.grid(True)
#     #         self.canvas_waveform.draw()
#     #     elif view_index == 1:
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(t, y_data)
#     #         self.ax_top.set_title("Waveform")
#     #         self.ax_top.set_xlabel("Time (s)")
#     #         self.ax_top.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         # Update the spectrum plot
#     #         self.ax_bottom.clear()
#     #         self.ax_bottom.plot(self.freqs, fft_mags)
#     #         self.ax_bottom.set_title("Spectrum")
#     #         self.ax_bottom.set_xlabel("Frequency (Hz)")
#     #         self.ax_bottom.set_ylabel(f"{y_label} RMS")
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()
#     #     elif view_index == 2:
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(t, y_data)
#     #         self.ax_top.set_title("Waveform (Channel 0)")
#     #         self.ax_top.set_xlabel("Time (s)")
#     #         self.ax_top.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         # Update the waveform plot for channel 1
#     #         self.ax_bottom.clear()
#     #         self.ax_bottom.plot(t, y_data1)
#     #         self.ax_bottom.set_title("Waveform (Channel 1)")
#     #         self.ax_bottom.set_xlabel("Time (s)")
#     #         self.ax_bottom.set_ylabel(y_label1)
#     #         margin1 = (max(y_data1) - min(y_data1)) * 0.1 or 0.2
#     #         self.ax_bottom.set_ylim(min(y_data1) - margin1, max(y_data1) + margin1)
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()

#     #     elif view_index == 3:
#     #         # Update the spectrum plot for channel 1
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(self.freqs, fft_mags)
#     #         self.ax_top.set_title("Spectrum (Channel 0)")
#     #         self.ax_top.set_xlabel("Frequency (Hz)")
#     #         self.ax_top.set_ylabel(f"{y_label} RMS")
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         # Update the spectrum plot for channel 2
#     #         self.ax_bottom.clear()
#     #         self.ax_bottom.plot(self.freqs1, fft_mags1)
#     #         self.ax_bottom.set_title("Spectrum (Channel 1)")
#     #         self.ax_bottom.set_xlabel("Frequency (Hz)")
#     #         self.ax_bottom.set_ylabel(f"{y_label1} RMS")
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()
#     #     # Update the readings
#     #     self.acc_input["input"].setText(f"{acc_peak0:.2f}")
#     #     self.vel_input["input"].setText(f"{vel_rms0:.2f}")
#     #     self.disp_input["input"].setText(f"{disp_pp0:.2f}")
#     #     self.freq_input["input"].setText(f"{dom_freq0:.2f}")
            


#     # def update_plot(self):
#     #     self.t, self.accel, self.velocity, self.displacement, acc_peak, acc_rms, vel_rms, disp_pp, dom_freq,fft_freqs,fft_mags,freqs_vel,fft_mags_vel,fft_freqs_disp,fft_mags_disp,rms_fft= self.daq.get_latest_waveform()
#     #     if len(self.t) == 0:
#     #         return


#     #     if self.selected_quantity == "Velocity":
#     #         y_data = self.velocity
#     #         N = len(y_data)
#     #         y_label = "Velocity (mm/s)"
#     #         pos_freqs = freqs_vel[:N // 2]
#     #         mask = pos_freqs <= self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_vel[mask]
#     #         self.fft_mags = fft_mags


#     #     elif self.selected_quantity == "Displacement":
#     #         y_data = self.displacement
#     #         y_label = "Displacement (μm)"
#     #         N2 = len(y_data)
#     #         pos_freqs = fft_freqs_disp[:N2 // 2]
#     #         mask = pos_freqs <= self.daq.fmax 
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags_disp[mask]
#     #         self.fft_mags = fft_mags
            
#     #     else:
#     #         y_data = self.accel
#     #         N1 = len(y_data)
#     #         y_label = "Acceleration (g)"
#     #         pos_freqs = fft_freqs[:N1 // 2]
#     #         mask = pos_freqs <=self.daq.fmax
#     #         self.freqs = pos_freqs[mask]
#     #         fft_mags = fft_mags[mask]
#     #         self.fft_mags = fft_mags

#     #     view_index = self.stacked_views.currentIndex()
#     #     if view_index == 0:
#     #         self.ax_waveform.clear()
#     #         self.ax_waveform.plot(self.t, y_data)
#     #         self.ax_waveform.set_title("Waveform")
#     #         self.ax_waveform.set_xlabel("Time (s)")
#     #         self.ax_waveform.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_waveform.grid(True)
#     #         self.canvas_waveform.draw()
#     #     else:
#     #         self.ax_top.clear()
#     #         self.ax_top.plot(self.t, y_data)
#     #         self.ax_top.set_title("Waveform")
#     #         self.ax_top.set_xlabel("Time (s)")
#     #         self.ax_top.set_ylabel(y_label)
#     #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
#     #         self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
#     #         self.ax_top.grid(True)
#     #         self.canvas_top.draw()

#     #         self.ax_bottom.clear()
#     #         #fft_result = np.fft.fft(y_data)
            
#     #         #freqs = np.fft.fftfreq(N, 1 / self.daq.actual_rate)
#     #         #fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
            
#     #         self.ax_bottom.plot(self.freqs, self.fft_mags)
#     #         self.ax_bottom.set_title("Spectrum")
#     #         self.ax_bottom.set_xlabel("Frequency (Hz)")
#     #         self.ax_bottom.set_ylabel(f"{y_label} RMS")
#     #         self.ax_bottom.grid(True)
#     #         self.canvas_bottom.draw()

#     #     self.acc_input["input"].setText(f"{acc_peak:.2f}")
#     #     self.vel_input["input"].setText(f"{rms_fft:.2f}")
#     #     self.disp_input["input"].setText(f"{disp_pp:.2f}")
#     #     self.freq_input["input"].setText(f"{dom_freq:.2f}")




























# import numpy as np
# import pandas as pd
# from daqhats import mcc172, OptionFlags, SourceType
# from scipy.fft import fft, fftfreq
# from scipy.signal import detrend, windows
# from endaq.calc.integrate import integrals
# from endaq.calc.filters import butterworth
# from endaq.calc.stats import rms


# class Mcc172Backend:
#     def __init__(self, board_num=0, sample_rate=11200, sensitivity=0.1, channel=None):
#         self.board_num = board_num
#         self.sample_rate = sample_rate
#         self.sensitivity = sensitivity
#         #self.channel = channel if channel is not None else self.auto_detect_channel()
#         self.board = mcc172(board_num)
#         self.buffer_size = None
#         self.actual_rate = None

#     # def auto_detect_channel(self):
#     #     for ch in [0, 1]:
#     #         self.board.iepe_config_write(ch, 1)
#     #         self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
#     #         _, actual_rate, _ = self.board.a_in_clock_config_read()
#     #         buffer_size = 2 ** int(np.floor(np.log2(actual_rate * 10)))
#     #         self.board.a_in_scan_start(1 << ch, buffer_size, OptionFlags.CONTINUOUS)
#     #         ch0_result,ch1_result = self.board.a_in_scan_read_numpy(-1, timeout=5.0)
#     #         self.board.a_in_scan_stop()
#     #         if ch0_result,ch1_result and np.any(ch0_result,ch1_result.data):
#     #             print(f"IEPE is connected to channel {ch}")
#     #             return ch
#     #     print("No IEPE sensor is connected")
#     #     return 0

#     def setup(self):
#         self.board.iepe_config_write(0, 1)
#         self.board.iepe_config_write(1, 1)
#         self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
#         _, self.actual_rate, _ = self.board.a_in_clock_config_read()

#         self.buffer_size = 65536
#         print(f"Actual sample rate: {self.actual_rate} Hz and buffer size: {self.buffer_size}")



#     def start_acquisition(self):
#         channel_mask = (1<< 0) | (1<<1)#e both channels 0 and 1
#         self.board.a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)

#     def stop_scan(self):
#         self.board.a_in_scan_stop()
#         self.board.a_in_scan_cleanup()
#         print("Scan stopped and cleaned up.")

#     # def read_data(self):
#     #     try:
#     #         result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
#     #     except Exception as e:
#     #         print(f"Error reading data: {e}")
#     #         return np.zeros(self.buffer_size), np.zeros(self.buffer_size)
#     #     if  result and result.data.size:
#     #         data = result.data
#     #         if data.ndim == 2 and data.shape[1] == 2:
#     #             ch0  = data[:, 0]
#     #             ch1  = data[:, 1]
#     #             return ch0,ch1
            
#     #         else:
#     #             print("Data format is not as expected. Expected two channels.")

#     #     return np.zeros(self.buffer_size), np.zeros(self.buffer_size)  # Return empty arrays if no data
#     #             # If data has two channels, average them

#     def read_data(self):
#         try:
#             result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
#         except Exception as e:
#             print(f"Error reading data: {e}")
#             return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

#         if result and result.data.size:
#             data = result.data

#             if data.ndim == 2 and data.shape[1] == 2:
#                 ch0 = data[:, 0]
#                 ch1 = data[:, 1]
#                 print(f"Channel 0 shape: {ch0.shape}, Channel 1 shape: {ch1.shape}")
#                 return ch0, ch1

#             elif data.ndim == 1:
#                 print("Only one channel active; interpreting as Channel 0.")
#                 half_length = data.size // 2
#                 return data[:half_length], data[half_length:]

#             else:
#                 print("Unexpected data shape:", data.shape)

#         print("No valid data received.")
#         return np.zeros(self.buffer_size), np.zeros(self.buffer_size)
# #     #     if data.ndim == 2:

#     # def read_data(self):
#     #     try:
#     #         result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
#     #     except Exception as e:
#     #         print(f"Error reading data: {e}")
#     #         return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

#     #     if result and result.data.size:
#     #         data = result.data

#     #         # Handle dual-channel properly
#     #         if data.ndim == 2:
#     #             ch0 = data[:, 0] if data.shape[1] > 0 else np.zeros(self.buffer_size)
#     #             ch1 = data[:, 1] if data.shape[1] > 1 else np.zeros(self.buffer_size)

#     #             # Simple signal presence detection: check if channel has non-zero variance
#     #             ch0_valid = not np.allclose(ch0, 0, atol=1e-4)
#     #             ch1_valid = not np.allclose(ch1, 0, atol=1e-4)

#     #             if ch0_valid and not ch1_valid:
#     #                 print("Sensor detected on Channel 0.")
#     #                 return ch0, np.zeros_like(ch0)
#     #             elif ch1_valid and not ch0_valid:
#     #                 print("Sensor detected on Channel 1.")
#     #                 return np.zeros_like(ch1), ch1
#     #             elif ch0_valid and ch1_valid:
#     #                 print("Sensors detected on both channels.")
#     #                 return ch0, ch1
#     #             else:
#     #                 print("No sensor signal detected.")
#     #                 return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

#     #         elif data.ndim == 1:
#     #             print("Only one channel of data returned; assuming it's Channel 0.")
#     #             return data, np.zeros_like(data)

#     #     print("No valid data received.")
#     #     return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

            
           

#     def analyze(self,signal_data,fmax=3600):
#         # Convert voltage to acceleration in g
#         acceleration_g = signal_data/self.sensitivity
#         acceleration_g = detrend(acceleration_g, type='linear')  # Remove trend
#         acceleration_m_s2 = acceleration_g * 9.80665  # Convert to m/s²

#         # Time setup
#         N = len(acceleration_m_s2)
#         print("Total No.of acceleration values",N)
#         T = 1 / self.actual_rate
#         time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
#         df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])

#         # Integrate to velocity and displacement
#         integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)
#         velocity_m_s = integrated[1]["acceleration"].to_numpy()
#         displacement_m = integrated[2]["acceleration"].to_numpy()

#         # Step 1: Detrend velocity
#         velocity_m_s = detrend(velocity_m_s, type='linear')

#         print("fmax:", fmax)

#         # Step 2: Bandpass filter for velocity
#         # Applied band pass filter using Butterworth filter
#         velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
#         velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
#         velocity_m_s = velocity_df['velocity'].to_numpy()

#         # Step 3: Apply Hanning window before FFT
#         velocity_mm_s = velocity_m_s * 1000  # Convert to mm/s
#         window = windows.hann(len(velocity_mm_s))
#         velocity_windowed = velocity_mm_s * window
#         window_correction = np.sqrt(np.mean(window ** 2))

#         # step 4: Detrend displacement  
#         displacement_m = detrend(displacement_m, type='linear')

#         # Step 5: Bandpass filter for displacement
#         displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
#         displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
#         displacement_m_s = displacement_df['displacement'].to_numpy()

#         displacement_um = displacement_m_s * 1e6  # Convert to µm
#         window_d = windows.hann(len(displacement_um))
#         displacement_windowed = displacement_um * window_d
#         window_correction_d = np.sqrt(np.mean(window_d ** 2))


        

#         N1 = len(velocity_mm_s)
#         N2 = len(displacement_um)
       

#         # RMS and metrics
#         vel_rms = rms(pd.Series(velocity_mm_s))
#         disp_pp = np.ptp(displacement_um)
#         acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
#         acc_peak = np.max(np.abs(acceleration_g))

#         # FFT for acceleration
#         fft_result = fft(acceleration_g)
#         freqs = fftfreq(N, T)
#         fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
#         pos_freqs = freqs[:N // 2]
#         dom_freq = pos_freqs[np.argmax(fft_mags)]

#         # FFT for velocity (with windowing)
#         fft_result_vel = fft(velocity_windowed)
#         freqs_vel = fftfreq(N1, T)
#         fft_mags_vel = (2.0 / N1) * np.abs(fft_result_vel[:N1 // 2])
#         pos_freqs_vel = freqs_vel[:N1 // 2]
#         #dom_freq_vel = pos_freqs_vel[np.argmax(fft_mags_vel)]

#         #FFT for displacement (with windowing)
#         fft_result_disp = fft(displacement_windowed)
#         freqs_disp = fftfreq(N2, T)
#         fft_mags_disp = (2.0 / N2) * np.abs(fft_result_disp[:N2 // 2])
#         pos_freqs_disp = freqs_disp[:N2 // 2]
#         #dom_freq_disp = pos_freqs_disp[np.argmax(fft_mags_disp)]


#         # Normalize RMS FFT for windowing loss
#         rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / window_correction
#         #velocity peak vaalue
#         vel_peak = np.max(np.abs(velocity_mm_s))
#         print("Velocity Peak (mm/s):", vel_peak)
#         vel_peak1= np.max(np.abs(velocity_windowed))
#         print("Velocity Peak (mm/s) with windowing:", vel_peak1)
#         # displacement peak value
#         disp_peak = np.max(np.abs(displacement_um))
#         disp_peak1 = np.max(np.abs(displacement_windowed))
#         print("Displacement Peak (um):", disp_peak)
#         print("Displacement Peak (um) with windowing:", disp_peak1)
#         print("Displacement Peak (um):", disp_pp)

#         print("Velocity RMS (mm/s):", vel_rms)
#         print("Dominant Frequency (Hz):", dom_freq)
#         print("RMS FFT:", rms_fft)

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
#             #"dom_freq_vel": dom_freq_vel,
#             "rms_fft": rms_fft
#         }

    

#     def get_latest_waveform(self,fmax=3600):
#         ch0_data,ch1_data = self.read_data()
#         if len(ch0_data) == 0:
#             print("No data received from MCC 172.")
#             return [], [], [], [], 0, 0, 0, 0, 0

#         ch0_result = self.analyze(ch0_data,fmax=fmax)
#         ch1_result=self.analyze(ch1_data,fmax=fmax)
       
#         return ch0_result,ch1_result



# # import numpy as np
# # import pandas as pd
# # from daqhats import mcc172, OptionFlags, SourceType
# # from scipy.fft import fft, fftfreq
# # from scipy.signal import detrend, windows
# # from endaq.calc.integrate import integrals
# # # from endaq.calc.filters import butterworth  # Temporarily bypassed
# # from endaq.calc.stats import rms

# # class Mcc172Backend:
# #     def __init__(self, board_num=0, sample_rate=11200, sensitivity=0.1):
# #         self.board_num = board_num
# #         self.sample_rate = sample_rate
# #         self.sensitivity = sensitivity
# #         self.board = mcc172(board_num)
# #         self.buffer_size = None
# #         self.actual_rate = None

# #     def setup(self):
# #         self.board.iepe_config_write(0, 1)
# #         self.board.iepe_config_write(1, 1)
# #         self.board.a_in_clock_config_write(SourceType.LOCAL, self.sample_rate)
# #         _, self.actual_rate, _ = self.board.a_in_clock_config_read()
# #         self.buffer_size =65536
# #         print(f"[Setup] Actual Sample Rate: {self.actual_rate} Hz, Buffer Size: {self.buffer_size}")

# #     def start_acquisition(self):
# #         channel_mask = (1 << 0) | (1 << 1)
# #         self.board.a_in_scan_start(channel_mask, self.buffer_size, OptionFlags.CONTINUOUS)

# #     def stop_scan(self):
# #         self.board.a_in_scan_stop()
# #         self.board.a_in_scan_cleanup()
# #         print("[Stop] Scan stopped and cleaned up.")

# #     def read_data(self):
# #         try:
# #             result = self.board.a_in_scan_read_numpy(self.buffer_size, timeout=5.0)
# #         except Exception as e:
# #             print(f"[Read Error] {e}")
# #             return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

# #         if result and result.data.size:
# #             data = result.data
# #             print(f"[Read] Raw shape: {data.shape}")
# #             if data.ndim == 2 and data.shape[1] == 2:
# #                 ch0 = data[:, 0]
# #                 ch1 = data[:, 1]
# #                 print(f"[Ch0] Mean: {np.mean(ch0):.4f}, Std: {np.std(ch0):.4f}")
# #                 print(f"[Ch1] Mean: {np.mean(ch1):.4f}, Std: {np.std(ch1):.4f}")
# #                 return ch0, ch1
# #             elif data.ndim == 1:
# #                 print("[Read] Only 1 channel active, assigning second channel to zero.")
# #                 return data, np.zeros_like(data)

# #         print("[Read] No valid data received.")
# #         return np.zeros(self.buffer_size), np.zeros(self.buffer_size)

# #     def analyze(self, signal_data, fmax=3600):
# #         acceleration_g = signal_data / self.sensitivity
# #         acceleration_g = detrend(acceleration_g, type='linear')
# #         print("[Analyze] First 10 Acc (g):", acceleration_g[:10])

# #         acceleration_m_s2 = acceleration_g * 9.80665
# #         N = len(acceleration_m_s2)
# #         T = 1 / self.actual_rate
# #         time_index = pd.timedelta_range(start='0s', periods=N, freq=pd.Timedelta(seconds=T))
# #         df = pd.DataFrame(acceleration_m_s2, index=time_index, columns=["acceleration"])

# #         integrated = integrals(df, n=2, zero="mean", tukey_percent=0.05)
# #         velocity_m_s = integrated[1]["acceleration"].to_numpy()
# #         displacement_m = integrated[2]["acceleration"].to_numpy()

# #         velocity_m_s = detrend(velocity_m_s, type='linear')
# #         # Bypassing filters for debug
# #         # velocity_df = pd.DataFrame(velocity_m_s, index=df.index, columns=['velocity'])
# #         # velocity_df = butterworth(velocity_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
# #         # velocity_m_s = velocity_df['velocity'].to_numpy()

# #         velocity_mm_s = velocity_m_s * 1000
# #         window = windows.hann(len(velocity_mm_s))
# #         velocity_windowed = velocity_mm_s * window
# #         window_correction = np.sqrt(np.mean(window ** 2))

# #         displacement_m = detrend(displacement_m, type='linear')
# #         # displacement_df = pd.DataFrame(displacement_m, index=df.index, columns=['displacement'])
# #         # displacement_df = butterworth(displacement_df, low_cutoff=3, high_cutoff=fmax, half_order=3)
# #         # displacement_m = displacement_df['displacement'].to_numpy()

# #         displacement_um = displacement_m * 1e6
# #         window_d = windows.hann(len(displacement_um))
# #         displacement_windowed = displacement_um * window_d

# #         N1 = len(velocity_mm_s)
# #         N2 = len(displacement_um)

# #         # RMS and Metrics
# #         vel_rms = rms(pd.Series(velocity_mm_s))
# #         disp_pp = np.ptp(displacement_um)
# #         acc_rms = np.sqrt(np.mean(acceleration_g ** 2))
# #         acc_peak = np.max(np.abs(acceleration_g))

# #         # FFT
# #         fft_result = fft(acceleration_g)
# #         freqs = fftfreq(N, T)
# #         fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
# #         pos_freqs = freqs[:N // 2]
# #         dom_freq = pos_freqs[np.argmax(fft_mags)]

# #         fft_result_vel = fft(velocity_windowed)
# #         freqs_vel = fftfreq(N1, T)
# #         fft_mags_vel = (2.0 / N1) * np.abs(fft_result_vel[:N1 // 2])
# #         pos_freqs_vel = freqs_vel[:N1 // 2]

# #         fft_result_disp = fft(displacement_windowed)
# #         freqs_disp = fftfreq(N2, T)
# #         fft_mags_disp = (2.0 / N2) * np.abs(fft_result_disp[:N2 // 2])
# #         pos_freqs_disp = freqs_disp[:N2 // 2]

# #         rms_fft = np.sqrt(np.sum((fft_mags_vel[1:] ** 2) / 2)) / window_correction

# #         print(f"[Analyze] Peak Acc (g): {acc_peak:.2f}, RMS Vel (mm/s): {vel_rms:.2f}, Disp (pp): {disp_pp:.2f}")
# #         print(f"[Analyze] Dom Freq: {dom_freq:.2f} Hz")

# #         return {
# #             "acceleration": acceleration_g,
# #             "velocity": velocity_mm_s,
# #             "displacement": displacement_um,
# #             "time": np.linspace(0, N * T, N, endpoint=False),
# #             "acc_peak": acc_peak,
# #             "acc_rms": acc_rms,
# #             "vel_rms": vel_rms,
# #             "disp_pp": disp_pp,
# #             "dom_freq": dom_freq,
# #             "fft_freqs": pos_freqs,
# #             "fft_mags": fft_mags,
# #             "freqs_vel": pos_freqs_vel,
# #             "fft_mags_vel": fft_mags_vel,
# #             "fft_freqs_disp": pos_freqs_disp,
# #             "fft_mags_disp": fft_mags_disp,
# #             "rms_fft": rms_fft
# #         }

# #     def get_latest_waveform(self, fmax=3600):
# #         ch0_data, ch1_data = self.read_data()
# #         if len(ch0_data) == 0:
# #             print("[Waveform] No data from Ch0.")
# #             return {}, {}

# #         ch0_result = self.analyze(ch0_data, fmax=fmax)
# #         ch1_result = self.analyze(ch1_data, fmax=fmax)
# #         return ch0_result, ch1_result




# from PyQt6.QtWidgets import (
#     QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
#     QSizePolicy, QMenu, QStackedWidget, QComboBox
# )
# from PyQt6.QtWidgets import (QDialog,QTableWidget,QTableWidgetItem,QVBoxLayout,QPushButton)
# from PyQt6.QtCore import Qt, QTimer, QDateTime
# from backend.mcc_backend import Mcc172Backend
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from pages.AnalysisParameters import AnalysisParameter
# from matplotlib.figure import Figure
# import numpy as np


# class WaveformPage(QWidget):
#     def __init__(self, main_window):
#         super().__init__()
#         self.main_window = main_window
#         #self.daq = Mcc172Backend(board_num=0, channel=1,sensitivity=0.1, sample_rate=51200)
#         #self.daq.setup()
#         # self.daq.start_acquisition()
#         # self.daq.auto_detect_channel()
#         self.selected_quantity = "Acceleration"
#         #self.is_running = False  # Measurement state tracker

#         self.setup_ui()
#         self.start_clock()
        

#         self.timer = QTimer()

#         self.timer.timeout.connect(self.update_plot)

#     def setup_ui(self):
#         self.main_layout = QVBoxLayout(self)
#         self.main_layout.setContentsMargins(0, 0, 0, 0)

#         nav_bar = QHBoxLayout()
#         title_label = QLabel("Waveform & Spectrum")
#         title_label.setStyleSheet("font-size: 1.5rem; font-weight: bold;")
#         title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

#         self.time_label = QLabel()
#         self.time_label.setStyleSheet("font-weight: bold; font-size: 1rem;")

#         nav_bar.addWidget(title_label)
#         nav_bar.addWidget(self.time_label)
#         self.main_layout.addLayout(nav_bar)

#         self.stacked_views = QStackedWidget()
#         self.default_view = self.build_readings_waveform_view()
#         self.dual_view = self.build_waveform_spectrum_view()
#         self.dual_waveform_view = self.build_dual_waveform_view()
#         self.dual_spectrum_view = self.build_dual_spectrum_view()

#         self.stacked_views.addWidget(self.default_view)
#         self.stacked_views.addWidget(self.dual_view)
#         self.stacked_views.addWidget(self.dual_waveform_view)
#         self.stacked_views.addWidget(self.dual_spectrum_view)
#         self.stacked_views.addWidget(self.build_readings_spectrum_view())
#         self.stacked_views.addWidget(self.build_dual_readings_view())
#         self.stacked_views.addWidget(self.trace_window_settings())
#         self.main_layout.addWidget(self.stacked_views)

#         bottom_buttons = QHBoxLayout()
#         for label in ["Traces", "Param", "Control", "Auto", "Cursor"]:
#             if label == "Traces":
#                 self.traces_button = QPushButton(label)
#                 self.traces_menu = QMenu()
#                 self.traces_menu.addAction("Readings + Waveform", lambda: self.switch_trace_mode(0))
#                 self.traces_menu.addAction("Waveform + Spectrum", lambda: self.switch_trace_mode(1))
#                 self.traces_menu.addAction("waveform(ch1 + ch2)", lambda: self.switch_trace_mode(2))
#                 self.traces_menu.addAction("Spectrum(ch1 + ch2)", lambda: self.switch_trace_mode(3))
#                 self.traces_menu.addAction("Readings + Spectrum)", lambda: self.switch_trace_mode(4))
#                 self.traces_menu.addAction("Dual Readings", lambda: self.switch_trace_mode(5))

#                 self.traces_menu.addAction("Trace + Window Settings", lambda: self.on_param_selected("Trace + Window Settings"))
#                 self.traces_button.setMenu(self.traces_menu)
#                 bottom_buttons.addWidget(self.traces_button)
#             elif label == "Param":
#                 self.params_button = QPushButton("Param")
#                 self.params_button.setStyleSheet("""
#                     background-color: #17a2b8;
#                     color: white;
#                     font-weight: bold;
#                     font-size: 1rem;
#                     padding: 8px;
#                     border-radius: 8px;
#                 """)
            
#                 self.params_Menu = QMenu(self.params_button)
#                 self.params_Menu.addAction("Analysis Parameters", self.analysis_parameter)
#                 self.params_Menu.addAction("Input channels", lambda: self.input_channels())
#                 self.params_Menu.addAction("Output channels", lambda: self.on_param_selected("Output Channels"))
#                 self.params_Menu.addAction("Tachometer", lambda: self.on_param_selected("Tachometer"))
#                 self.params_Menu.addAction("Display Preferences", lambda: self.on_param_selected("Display Preferences"))
#                 self.params_Menu.addAction("view Live Signals", lambda: self.on_param_selected("view Live Signals"))
#                 self.params_button.setMenu(self.params_Menu)
#                 bottom_buttons.addWidget(self.params_button)
            
#             else:
#                 btn = QPushButton(label)
#                 btn.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold;")
#                 bottom_buttons.addWidget(btn)

#         # Start Meas. and Stop Meas. buttons
#         self.start_button = QPushButton("Start Meas.")
#         self.start_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
#         self.start_button.clicked.connect(self.start_measurement)
#         bottom_buttons.addWidget(self.start_button)

#         self.stop_button = QPushButton("Stop Meas.")
#         self.stop_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
#         self.stop_button.clicked.connect(self.stop_measurement)
#         self.stop_button.setVisible(False)
#         bottom_buttons.addWidget(self.stop_button)

#         self.main_layout.addLayout(bottom_buttons)

#     def on_param_selected(self, option):
#         pass

#     def analysis_parameter(self):
#         self.popup = AnalysisParameter(self)
#         self.popup.show()
    
#     def input_channels(self):
#         self.popup_input_channels = InputChannelsDialog(self)
#         if self.popup_input_channels.exec():  # Waits until Apply or Close
#             self.channel_configs = self.popup_input_channels.channel_config
#             print("Received channel configuration in WaveformPage:")
#             for config in self.channel_configs:
#                 print(config)

#     def get_enabled_channels(self):
#         enabled_channels = []
#         if not hasattr(self, 'channel_configs'):
#             return enabled_channels
        
#         for config in self.channel_configs:
#             if isinstance(config, dict) and "ch" in config and "sensitivity" in config:
#                 try:
#                     ch_num = int(config["ch"])
#                     board_num = (ch_num - 1) // 2
#                     channel = (ch_num - 1) % 2
#                     sensitivity_str = config["sensitivity"]
#                     sensitivity = float(sensitivity_str.split(" ")[0])  # Extract numeric part
#                     enabled_channels.append({
#                         "board_num": board_num,
#                         "channel": channel,
#                         "sensitivity": sensitivity,
#                     })
#                 except Exception as e:
#                     print(f"Error processing channel config {config}: {e}")
#         return enabled_channels
                



#     def build_readings_waveform_view(self):
#         widget = QWidget()
#         layout = QVBoxLayout(widget)

#         readings_layout = QHBoxLayout()
#         self.acc_input = self.create_reading_box("Acc:", "g (peak)")
#         self.vel_input = self.create_reading_box("Vel:", "mm/s (RMS)")
#         self.disp_input = self.create_reading_box("Disp:", "µm (P-P)")
#         self.freq_input = self.create_reading_box("Freq:", "Hz")

#         for box in [self.acc_input, self.vel_input, self.disp_input, self.freq_input]:
#             readings_layout.addLayout(box["layout"])
#         layout.addLayout(readings_layout)

#         self.figure_waveform = Figure(figsize=(6, 3))
#         self.canvas_waveform = FigureCanvas(self.figure_waveform)
#         self.ax_waveform = self.figure_waveform.add_subplot(111)
#         self.canvas_waveform.mpl_connect("button_press_event", self.on_waveform_click)
#         layout.addWidget(self.canvas_waveform)
#         return widget

#     def build_waveform_spectrum_view(self):
#         widget = QWidget()
#         layout = QVBoxLayout(widget)

#         self.figure_top = Figure(figsize=(6, 3))
#         self.canvas_top = FigureCanvas(self.figure_top)
#         self.ax_top = self.figure_top.add_subplot(111)
#         self.canvas_top.mpl_connect("button_press_event", self.on_waveform_click)
#         layout.addWidget(self.canvas_top)

#         self.figure_bottom = Figure(figsize=(6, 3))
#         self.canvas_bottom = FigureCanvas(self.figure_bottom)
#         self.ax_bottom = self.figure_bottom.add_subplot(111)
#         self.canvas_bottom.mpl_connect("button_press_event", self.on_fft_click)
#         layout.addWidget(self.canvas_bottom)

#         return widget
#     def build_dual_waveform_view(self):
#         widget = QWidget()
#         layout = QVBoxLayout(widget)
#         self.fig_waveform_ch = Figure()
#         self.canvas_waveform_ch = FigureCanvas(self.fig_waveform_ch)
#         self.ax_ch1_waveform = self.fig_waveform_ch.add_subplot(211)
#         self.ax_ch2_waveform = self.fig_waveform_ch.add_subplot(212)
#         layout.addWidget(self.canvas_waveform_ch)
#         return widget
#     def build_dual_spectrum_view(self):
#         widget = QWidget()
#         layout = QVBoxLayout(widget)
#         self.fig_fft_ch = Figure()
#         self.canvas_fft_ch = FigureCanvas(self.fig_fft_ch)
#         self.ax_ch1_spec = self.fig_fft_ch.add_subplot(211)
#         self.ax_ch2_spec = self.fig_fft_ch.add_subplot(212)
#         layout.addWidget(self.canvas_fft_ch)
#         return widget
#     def build_readings_spectrum_view(self):
#         widget = QWidget()
#         layout = QVBoxLayout(widget)
#         readings_layout = QHBoxLayout()
#         self.acc_input = self.create_reading_box("Acc:", "g (peak)")
#         self.vel_input = self.create_reading_box("Vel:", "mm/s (RMS)")
#         self.disp_input = self.create_reading_box("Disp:", "µm (P-P)")
#         self.freq_input = self.create_reading_box("Freq:", "Hz")
#         for box in [self.acc_input, self.vel_input, self.disp_input, self.freq_input]:
#             readings_layout.addLayout(box["layout"])
#         layout.addLayout(readings_layout)
#         self.figure_spectrum = Figure(figsize=(6, 3))
#         self.canvas_spectrum = FigureCanvas(self.figure_spectrum)
#         self.ax_spectrum = self.figure_spectrum.add_subplot(111)
#         self.canvas_spectrum.mpl_connect("button_press_event", self.on_fft_click)
#         layout.addWidget(self.canvas_spectrum)

#         return widget
    
#     def build_dual_readings_view(self):
#         widget = QWidget()
#         layout = QVBoxLayout(widget)

#         readings_layout = QHBoxLayout()
#         # Channel 1 readings
#         ch1_layout = QVBoxLayout()
#         ch1_label = QLabel("CH1")
#         ch1_label.setStyleSheet("font-weight: bold; font-size: 1.1rem; color: #007bff;")
#         ch1_layout.addWidget(ch1_label)
#         ch1_acc = self.create_reading_box("Acc:", "g (peak)")
#         ch1_vel = self.create_reading_box("Vel:", "mm/s (RMS)")
#         ch1_disp = self.create_reading_box("Disp:", "µm (P-P)")
#         ch1_freq = self.create_reading_box("Freq:", "Hz")
#         for box in [ch1_acc, ch1_vel, ch1_disp, ch1_freq]:
#             ch1_layout.addLayout(box["layout"])

#         # Channel 2 readings
#         ch2_layout = QVBoxLayout()
#         ch2_label = QLabel("CH2")
#         ch2_label.setStyleSheet("font-weight: bold; font-size: 1.1rem; color: #28a745;")
#         ch2_layout.addWidget(ch2_label)
#         ch2_acc = self.create_reading_box("Acc:", "g (peak)")
#         ch2_vel = self.create_reading_box("Vel:", "mm/s (RMS)")
#         ch2_disp = self.create_reading_box("Disp:", "µm (P-P)")
#         ch2_freq = self.create_reading_box("Freq:", "Hz")
#         for box in [ch2_acc, ch2_vel, ch2_disp, ch2_freq]:
#             ch2_layout.addLayout(box["layout"])

#         readings_layout.addLayout(ch1_layout)
#         readings_layout.addSpacing(30)
#         readings_layout.addLayout(ch2_layout)
#         layout.addLayout(readings_layout)

#         widget.ch1_readings = {
#             "acc": ch1_acc["input"],
#             "vel": ch1_vel["input"],
#             "disp": ch1_disp["input"],
#             "freq": ch1_freq["input"],
#         }
#         widget.ch2_readings = {
#             "acc": ch2_acc["input"],
#             "vel": ch2_vel["input"],
#             "disp": ch2_disp["input"],
#             "freq": ch2_freq["input"],
#         }

#         return widget
    
#     # Add navigation arrows and channel selection above the trace views.
#     # This will allow switching between traces and selecting channels for top/bottom views.

#     def trace_window_settings(self):
#         # Navigation bar for traces (left/right arrows)
#         nav_layout = QHBoxLayout()
#         self.prev_trace_btn = QPushButton("←")
#         self.next_trace_btn = QPushButton("→")
#         self.prev_trace_btn.setFixedWidth(40)
#         self.next_trace_btn.setFixedWidth(40)
#         self.prev_trace_btn.clicked.connect(self.prev_trace)
#         self.next_trace_btn.clicked.connect(self.next_trace)
#         nav_layout.addWidget(self.prev_trace_btn)
#         nav_layout.addWidget(QLabel("Select Trace View:"))
#         self.trace_combo = QComboBox()
#         self.trace_combo.addItems([
#         "Readings + Waveform",
#         "Waveform + Spectrum",
#         "Waveform (ch1 + ch2)",
#         "Spectrum (ch1 + ch2)",
#         "Readings + Spectrum",
#         "Dual Readings"
#         ])
#         self.trace_combo.currentIndexChanged.connect(self.switch_trace_mode)
#         nav_layout.addWidget(self.trace_combo)
#         nav_layout.addWidget(self.next_trace_btn)

#         # Channel selection for top and bottom views
#         ch_layout = QHBoxLayout()
#         ch_layout.addWidget(QLabel("Top View Channel:"))
#         self.top_channel_combo = QComboBox()
#         self.top_channel_combo.addItems([f"CH{i+1}" for i in range(6)])
#         self.top_channel_combo.currentIndexChanged.connect(self.update_top_channel)
#         ch_layout.addWidget(self.top_channel_combo)

#         ch_layout.addSpacing(20)
#         ch_layout.addWidget(QLabel("Bottom View Channel:"))
#         self.bottom_channel_combo = QComboBox()
#         self.bottom_channel_combo.addItems([f"CH{i+1}" for i in range(6)])
#         self.bottom_channel_combo.currentIndexChanged.connect(self.update_bottom_channel)
#         ch_layout.addWidget(self.bottom_channel_combo)

#         # Add these layouts to the main layout above the stacked views
#         self.main_layout.insertLayout(1, nav_layout)
#         self.main_layout.insertLayout(2, ch_layout)

#         # Store current trace/channel selections
#         self.current_trace_index = 0
#         self.current_top_channel = 0
#         self.current_bottom_channel = 0

#     def prev_trace(self):
#         idx = (self.trace_combo.currentIndex() - 1) % self.trace_combo.count()
#         self.trace_combo.setCurrentIndex(idx)

#     def next_trace(self):
#         idx = (self.trace_combo.currentIndex() + 1) % self.trace_combo.count()
#         self.trace_combo.setCurrentIndex(idx)

#     def update_top_channel(self, idx):
#         self.current_top_channel = idx
#         # Update the plot for the selected top channel
#         self.update_plot()

#     def update_bottom_channel(self, idx):
#         self.current_bottom_channel = idx
#         # Update the plot for the selected bottom channel
#         self.update_plot()


    

#     def create_reading_box(self, label_text, unit_text):
#         layout = QVBoxLayout()
#         label = QLabel(label_text)
#         label.setStyleSheet("font-weight: bold; font-size: 1rem;")
#         input_field = QLineEdit("0")
#         input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         input_field.setReadOnly(True)
#         input_field.setFixedWidth(80)
#         unit_label = QLabel(unit_text)
#         unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         unit_label.setStyleSheet("font-size: 0.9rem; color: #666;")
#         layout.addWidget(label)
#         layout.addWidget(input_field)
#         layout.addWidget(unit_label)
#         return {"layout": layout, "input": input_field}

#     def start_clock(self):
#         self.clock = QTimer()
#         self.clock.timeout.connect(self.update_time)
#         self.clock.start(1000)
#         self.update_time()

#     def update_time(self):
#         now = QDateTime.currentDateTime()
#         self.time_label.setText(now.toString("HH:mm:ss"))

#     def switch_trace_mode(self, index):
        
       
#         self.stacked_views.setCurrentIndex(index)
#     """
#     def start_measurement(self):
#         if not self.is_running:
#             self.daq.start_acquisition()
#             self.timer.start(1000)
#             self.is_running = True
#             self.start_button.setVisible(False)
#             self.stop_button.setVisible(True)
#             print("️Measurement started.")
#         else:
#             print("Measurement is already running.")
# """ 
#     def start_measurement(self):
#         enabled_channels = self.get_enabled_channels()
#         if not enabled_channels:
#             print("No enabled channels found. Please configure input channels.")
#             return
#         # self.daq = Mcc172Backend(board_num=enabled_channels[0]["board_num"],
#         #                          channel=enabled_channels[0]["channel"],
#         #                          sensitivity=enabled_channels[0]["sensitivity"],
#         #   
#         self.daq = Mcc172Backend(
#             channel_configs=enabled_channels,
#             selected_channels= [0, 1],  # Assuming we want to use the first two channels
#             board_num=0,
#             sensitivity=0.1,  # Example sensitivity, adjust as needed
#             buffer_size=51200,  # Example buffer size, adjust as needed 
#             sample_rate=51200
#         )
#         self.daq.setup()
#         self.daq.start_acquisition()
#         self.timer.start(1000)
#         self.start_button.setVisible(False)
#         self.stop_button.setVisible(True)


#     def stop_measurement(self):
#         self.daq.stop_scan()
#         self.timer.stop()
#         #self.is_running = False
#         self.start_button.setVisible(True)
#         self.stop_button.setVisible(False)
#         print("Measurement stopped.")

#     def update_plot(self):
#         results = self.daq.get_latest_waveform()

#         if not results or len(results) < 2:
#             print("❌ Insufficient data to plot.")
#             return

#         try:
#             (
#                 t1, acc1, vel1, disp1, acc_peak1, acc_rms1, vel_rms1, disp_pp1, dom_freq1,
#                 fft_freqs1, fft_mags1, fft_freqs_vel1, fft_mags_vel1, fft_freqs_disp1, fft_mags_disp1, rms_fft1
#             ) = results[0]

#             (
#                 t2, acc2, vel2, disp2, acc_peak2, acc_rms2, vel_rms2, disp_pp2, dom_freq2,
#                 fft_freqs2, fft_mags2, fft_freqs_vel2, fft_mags_vel2, fft_freqs_disp2, fft_mags_disp2, rms_fft2
#             ) = results[1]

#             if self.selected_quantity == "Acceleration":
#                 y1, y2 = acc1, acc2
#                 fft_y1, fft_y2 = fft_mags1, fft_mags2
#                 fft_x1, fft_x2 = fft_freqs1, fft_freqs2
#             elif self.selected_quantity == "Velocity":
#                 y1, y2 = vel1, vel2
#                 fft_y1, fft_y2 = fft_mags_vel1, fft_mags_vel2
#                 fft_x1, fft_x2 = fft_freqs_vel1, fft_freqs_vel2
#             elif self.selected_quantity == "Displacement":
#                 y1, y2 = disp1, disp2
#                 fft_y1, fft_y2 = fft_mags_disp1, fft_mags_disp2
#                 fft_x1, fft_x2 = fft_freqs_disp1, fft_freqs_disp2
#             else:
#                 print("❌ Unknown Measurement Quantity selected.")
#                 return

#             self.ax1.clear()
#             if t1 and y1 and len(t1) == len(y1):
#                 self.ax1.plot(t1, y1)
#             else:
#                 self.ax1.text(0.5, 0.5, "No valid data", ha='center', va='center')
#             self.ax1.set_title("Top Trace")
#             self.ax1.set_xlabel("Time (s)")
#             self.ax1.set_ylabel(self.selected_quantity)

#             self.ax2.clear()
#             if t2 and y2 and len(t2) == len(y2):
#                 self.ax2.plot(t2, y2)
#             else:
#                 self.ax2.text(0.5, 0.5, "No valid data", ha='center', va='center')
#             self.ax2.set_title("Bottom Trace")
#             self.ax2.set_xlabel("Time (s)")
#             self.ax2.set_ylabel(self.selected_quantity)

#             self.ax3.clear()
#             if fft_x1 and fft_y1 and len(fft_x1) == len(fft_y1):
#                 self.ax3.plot(fft_x1, fft_y1)
#             else:
#                 self.ax3.text(0.5, 0.5, "No FFT data", ha='center', va='center')
#             self.ax3.set_title("Top Trace FFT")
#             self.ax3.set_xlabel("Frequency (Hz)")
#             self.ax3.set_ylabel(f"{self.selected_quantity} Spectrum")

#             self.ax4.clear()
#             if fft_x2 and fft_y2 and len(fft_x2) == len(fft_y2):
#                 self.ax4.plot(fft_x2, fft_y2)
#             else:
#                 self.ax4.text(0.5, 0.5, "No FFT data", ha='center', va='center')
#             self.ax4.set_title("Bottom Trace FFT")
#             self.ax4.set_xlabel("Frequency (Hz)")
#             self.ax4.set_ylabel(f"{self.selected_quantity} Spectrum")

#             self.canvas.draw()
        
#         except Exception as e:
#             print(f"❌ Exception during plotting: {e}")

#     def on_waveform_click(self, event):
#         if not hasattr(self, 't') or event.inaxes not in [self.ax_waveform, self.ax_top]:
#             return
#         x = event.xdata
#         idx = min(range(len(self.t)), key=lambda i: abs(self.t[i] - x))
#         t_val = self.t[idx]
#         y_val = self.accel[idx]
#         event.inaxes.annotate(f"({t_val:.3f}s, {y_val:.3f}g)", (t_val, y_val),
#                               textcoords="offset points", xytext=(10, 10),
#                               arrowprops=dict(arrowstyle="->", color='red'),
#                               fontsize=9, color='red')
#         event.canvas.draw()

#     def on_fft_click(self, event):
#         if not hasattr(self, 'freqs') or event.inaxes != self.ax_bottom:
#             return
#         x = event.xdata
#         idx = min(range(len(self.freqs)), key=lambda i: abs(self.freqs[i] - x))
#         f_val = self.freqs[idx]
#         mag = self.fft_mags[idx]
#         self.ax_bottom.annotate(f"({f_val:.1f} Hz, {mag:.3f})", (f_val, mag),
#                                 textcoords="offset points", xytext=(10, 10),
#                                 arrowprops=dict(arrowstyle="->", color='green'),
#                                 fontsize=9, color='green')
#         self.canvas_bottom.draw()
# class InputChannelsDialog(QDialog):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Input Channel and Sensor Configuration")
#         self.setMinimumSize(600, 300)

#         self.table = QTableWidget(6, 5)
#         self.table.setHorizontalHeaderLabels(["Ch.", "Sensitivity", "Input Mode", "HP Fltr"])
#         self.table.verticalHeader().setVisible(False)

#         # Sample data
#         default_data = [
#             ["1", "0.1 g", "IEPE", "1Hz"],
#             ["2", "0.1 g", "IEPE", "1Hz"],
#             ["3", "0.1 g", "IEPE", "1Hz"],
#             ["4", "0.1 g", "IEPE", "1Hz"],
#             ["5", "0.1 g", "IEPE", "1Hz"],
#             ["6", "0.1 g", "IEPE", "1Hz"]
#         ]

#         # Fill table
#         # for row, row_data in enumerate(default_data):
#         #     for col, value in enumerate(row_data):
#         #         if col == 4:  # Status column → dropdown
#         #             combo = QComboBox()
#         #             combo.addItems(["ON", "OFF"])
#         #             combo.setCurrentText(value)
#         #             self.table.setCellWidget(row, col, combo)
#         #         else:
#         #             item = QTableWidgetItem(value)
#         #             if col == 0:
#         #                 item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)  # Make Ch. read-only
#         #             self.table.setItem(row, col, item)

#         layout = QVBoxLayout()
#         layout.addWidget(self.table)

#         # Apply Button
#         apply_btn = QPushButton("Apply")
#         apply_btn.clicked.connect(self.apply_and_close)
#         layout.addWidget(apply_btn)

#         self.setLayout(layout)

#     def apply_and_close(self):
#         self.channel_config = []
#         for row in range(self.table.rowCount()):
#             ch = self.table.item(row, 0).text()
#             sensitivity = self.table.item(row, 1).text()
#             input_mode = self.table.item(row, 2).text()
#             hp_filter = self.table.item(row, 3).text()
            
#             self.channel_config.append({
#                 "ch": int(ch),
#                 "sensitivity": sensitivity,
#                 "input_mode": input_mode,
#                 "hp_filter": hp_filter,
                
#             })

#         print("Collected Channel Configurations:")
#         for config in self.channel_config:
#             print(config)

#         self.accept()  # Close dialog



