
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QSizePolicy, QMenu, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from backend.mcc_backend import Mcc172Backend
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pages.AnalysisParameters import AnalysisParameter
from matplotlib.figure import Figure
import numpy as np


class WaveformPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.daq = Mcc172Backend(board_num=0, sensitivity=0.1, sample_rate=11200)
        self.daq.setup()
        # self.daq.start_acquisition()
        # self.daq.auto_detect_channel()
        self.selected_quantity = "Acceleration"
        self.selected_fmax = 3600
        #self.is_running = False  # Measurement state tracker

        self.setup_ui()
        self.start_clock()
        

        self.timer = QTimer()

        self.timer.timeout.connect(self.update_plot)

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        nav_bar = QHBoxLayout()
        title_label = QLabel("Waveform & Spectrum")
        title_label.setStyleSheet("font-size: 1.5rem; font-weight: bold;")
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-weight: bold; font-size: 1rem;")

        nav_bar.addWidget(title_label)
        nav_bar.addWidget(self.time_label)
        self.main_layout.addLayout(nav_bar)

        self.stacked_views = QStackedWidget()
        self.default_view = self.build_readings_waveform_view()
        self.dual_view = self.build_waveform_spectrum_view()
        self.channel_waveform_view = self.build_dual_waveform_view()
        self.channel_spectrum_view = self.build_dual_spectrum_view()



        self.stacked_views.addWidget(self.default_view)
        self.stacked_views.addWidget(self.dual_view)
        self.stacked_views.addWidget(self.channel_waveform_view)
        self.stacked_views.addWidget(self.channel_spectrum_view)
        self.main_layout.addWidget(self.stacked_views)

        bottom_buttons = QHBoxLayout()
        for label in ["Traces", "Param", "Control", "Auto", "Cursor"]:
            if label == "Traces":
                self.traces_button = QPushButton(label)
                self.traces_menu = QMenu()
                self.traces_menu.addAction("Readings + Waveform", lambda: self.switch_trace_mode(0))
                self.traces_menu.addAction("Waveform + Spectrum", lambda: self.switch_trace_mode(1))
                self.traces_menu.addAction("Waveform(ch1 + ch2)", lambda: self.switch_trace_mode(2))
                self.traces_menu.addAction("Spectrum(ch1 + ch2)", lambda: self.switch_trace_mode(3))
                self.traces_button.setMenu(self.traces_menu)
                bottom_buttons.addWidget(self.traces_button)
            elif label == "Param":
                self.params_button = QPushButton("Param")
                self.params_button.setStyleSheet("""
                    background-color: #17a2b8;
                    color: white;
                    font-weight: bold;
                    font-size: 1rem;
                    padding: 8px;
                    border-radius: 8px;
                """)
                self.params_Menu = QMenu(self.params_button)
                self.params_Menu.addAction("Analysis Parameters", self.analysis_parameter)
                self.params_Menu.addAction("Input channels", lambda: self.on_param_selected("Input Channels"))
                self.params_Menu.addAction("Output channels", lambda: self.on_param_selected("Output Channels"))
                self.params_Menu.addAction("Tachometer", lambda: self.on_param_selected("Tachometer"))
                self.params_Menu.addAction("Display Preferences", lambda: self.on_param_selected("Display Preferences"))
                self.params_Menu.addAction("view Live Signals", lambda: self.on_param_selected("view Live Signals"))
                self.params_button.setMenu(self.params_Menu)
                bottom_buttons.addWidget(self.params_button)
            else:
                btn = QPushButton(label)
                btn.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold;")
                bottom_buttons.addWidget(btn)

        # Start Meas. and Stop Meas. buttons
        self.start_button = QPushButton("Start Meas.")
        self.start_button.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self.start_measurement)
        bottom_buttons.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Meas.")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.stop_button.clicked.connect(self.stop_measurement)
        self.stop_button.setVisible(False)
        bottom_buttons.addWidget(self.stop_button)

        self.main_layout.addLayout(bottom_buttons)

    def on_param_selected(self, option):
        pass

    def analysis_parameter(self):
        self.popup = AnalysisParameter(self)
        self.popup.show()

    def build_readings_waveform_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        readings_layout = QHBoxLayout()
        self.acc_input = self.create_reading_box("Acc:", "g (peak)")
        self.vel_input = self.create_reading_box("Vel:", "mm/s (RMS)")
        self.disp_input = self.create_reading_box("Disp:", "µm (P-P)")
        self.freq_input = self.create_reading_box("Freq:", "Hz")

        for box in [self.acc_input, self.vel_input, self.disp_input, self.freq_input]:
            readings_layout.addLayout(box["layout"])
        layout.addLayout(readings_layout)

        self.figure_waveform = Figure(figsize=(6, 3))
        self.canvas_waveform = FigureCanvas(self.figure_waveform)
        self.ax_waveform = self.figure_waveform.add_subplot(111)
        self.canvas_waveform.mpl_connect("button_press_event", self.on_waveform_click)
        layout.addWidget(self.canvas_waveform)
        return widget

    def build_waveform_spectrum_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.figure_top = Figure(figsize=(6, 3))
        self.canvas_top = FigureCanvas(self.figure_top)
        self.ax_top = self.figure_top.add_subplot(111)
        self.canvas_top.mpl_connect("button_press_event", self.on_waveform_click)
        layout.addWidget(self.canvas_top)

        self.figure_bottom = Figure(figsize=(6, 3))
        self.canvas_bottom = FigureCanvas(self.figure_bottom)
        self.ax_bottom = self.figure_bottom.add_subplot(111)
        self.canvas_bottom.mpl_connect("button_press_event", self.on_fft_click)
        layout.addWidget(self.canvas_bottom)

        return widget
    def build_dual_waveform_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.fig_wave_ch = Figure()
        self.canvas_wave_ch = FigureCanvas(self.fig_wave_ch)
        self.ax_ch1 = self.fig_wave_ch.add_subplot(211)
        self.ax_ch2 = self.fig_wave_ch.add_subplot(212)
        layout.addWidget(self.canvas_wave_ch)
        return widget
    
    def build_dual_spectrum_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.fig_fft_ch = Figure()
        self.canvas_fft_ch = FigureCanvas(self.fig_fft_ch)
        self.ax_ch1_spec = self.fig_fft_ch.add_subplot(211)
        self.ax_ch2_spec = self.fig_fft_ch.add_subplot(212)
        layout.addWidget(self.canvas_fft_ch)
        return widget

    def create_reading_box(self, label_text, unit_text):
        layout = QVBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("font-weight: bold; font-size: 1rem;")
        input_field = QLineEdit("0")
        input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_field.setReadOnly(True)
        input_field.setFixedWidth(80)
        unit_label = QLabel(unit_text)
        unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit_label.setStyleSheet("font-size: 0.9rem; color: #666;")
        layout.addWidget(label)
        layout.addWidget(input_field)
        layout.addWidget(unit_label)
        return {"layout": layout, "input": input_field}

    def start_clock(self):
        self.clock = QTimer()
        self.clock.timeout.connect(self.update_time)
        self.clock.start(1000)
        self.update_time()

    def update_time(self):
        now = QDateTime.currentDateTime()
        self.time_label.setText(now.toString("HH:mm:ss"))

    def switch_trace_mode(self, index):
        
       
        self.stacked_views.setCurrentIndex(index)
    """
    def start_measurement(self):
        if not self.is_running:
            self.daq.start_acquisition()
            self.timer.start(1000)
            self.is_running = True
            self.start_button.setVisible(False)
            self.stop_button.setVisible(True)
            print("️Measurement started.")
        else:
            print("Measurement is already running.")
""" 
    def start_measurement(self):
        self.daq.start_acquisition()
        self.timer.start(1000)
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)


    def stop_measurement(self):
        self.daq.stop_scan()
        self.timer.stop()
        #self.is_running = False
        self.start_button.setVisible(True)
        self.stop_button.setVisible(False)
        print("Measurement stopped.")

    # def update_plot(self):
    #     self.t, self.accel, self.velocity, self.displacement, acc_peak, acc_rms, vel_rms, disp_pp, dom_freq,fft_freqs,fft_mags,freqs_vel,fft_mags_vel,fft_freqs_disp,fft_mags_disp,rms_fft= self.daq.get_latest_waveform(fmax=self.selected_fmax)
    #     if len(self.t) == 0:
    #         return


    #     if self.selected_quantity == "Velocity":
    #         y_data = self.velocity
    #         N = len(y_data)
    #         y_label = "Velocity (mm/s)"
    #         pos_freqs = freqs_vel[:N // 2]
    #         mask = pos_freqs <= 500
    #         self.freqs = pos_freqs[mask]
    #         fft_mags = fft_mags_vel[mask]
    #         self.fft_mags = fft_mags


    #     elif self.selected_quantity == "Displacement":
    #         y_data = self.displacement
    #         y_label = "Displacement (μm)"
    #         N2 = len(y_data)
    #         pos_freqs = fft_freqs_disp[:N2 // 2]
    #         mask = pos_freqs <= 500 
    #         self.freqs = pos_freqs[mask]
    #         fft_mags = fft_mags_disp[mask]
    #         self.fft_mags = fft_mags
            
    #     else:
    #         y_data = self.accel
    #         N1 = len(y_data)
    #         y_label = "Acceleration (g)"
    #         pos_freqs = fft_freqs[:N1 // 2]
    #         mask = pos_freqs <=500
    #         self.freqs = pos_freqs[mask]
    #         fft_mags = fft_mags[mask]
    #         self.fft_mags = fft_mags

    #     view_index = self.stacked_views.currentIndex()
    #     if view_index == 0:
    #         self.ax_waveform.clear()
    #         self.ax_waveform.plot(self.t, y_data)
    #         self.ax_waveform.set_title("Waveform")
    #         self.ax_waveform.set_xlabel("Time (s)")
    #         self.ax_waveform.set_ylabel(y_label)
    #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
    #         self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
    #         self.ax_waveform.grid(True)
    #         self.canvas_waveform.draw()
    #     else:
    #         self.ax_top.clear()
    #         self.ax_top.plot(self.t, y_data)
    #         self.ax_top.set_title("Waveform")
    #         self.ax_top.set_xlabel("Time (s)")
    #         self.ax_top.set_ylabel(y_label)
    #         margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
    #         self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
    #         self.ax_top.grid(True)
    #         self.canvas_top.draw()

    #         self.ax_bottom.clear()
    #         #fft_result = np.fft.fft(y_data)
            
    #         #freqs = np.fft.fftfreq(N, 1 / self.daq.actual_rate)
    #         #fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
            
    #         self.ax_bottom.plot(self.freqs, self.fft_mags)
    #         self.ax_bottom.set_title("Spectrum")
    #         self.ax_bottom.set_xlabel("Frequency (Hz)")
    #         self.ax_bottom.set_ylabel(f"{y_label} RMS")
    #         self.ax_bottom.grid(True)
    #         self.canvas_bottom.draw()

    #     self.acc_input["input"].setText(f"{acc_peak:.2f}")
    #     self.vel_input["input"].setText(f"{rms_fft:.2f}")
    #     self.disp_input["input"].setText(f"{disp_pp:.2f}")
    #     self.freq_input["input"].setText(f"{dom_freq:.2f}")

    def update_plot(self):
        view_index = self.stacked_views.currentIndex()
        ch0_result, ch1_result = self.daq.get_latest_waveform(fmax=self.selected_fmax)

        if not ch0_result or "time" not in ch0_result:
            print("No valid Ch0 data.")
            return

        # Determine which data to use for single-channel views (Ch0)
        if self.selected_quantity == "Velocity":
            y_data = ch0_result["velocity"]
            y_label = "Velocity (mm/s)"
            fft_freqs = ch0_result["freqs_vel"]
            fft_mags = ch0_result["fft_mags_vel"]
        elif self.selected_quantity == "Displacement":
            y_data = ch0_result["displacement"]
            y_label = "Displacement (μm)"
            fft_freqs = ch0_result["fft_freqs_disp"]
            fft_mags = ch0_result["fft_mags_disp"]
        else:
            y_data = ch0_result["acceleration"]
            y_label = "Acceleration (g)"
            fft_freqs = ch0_result["fft_freqs"]
            fft_mags = ch0_result["fft_mags"]

        if view_index == 0:
            # Readings + Waveform
            self.ax_waveform.clear()
            self.ax_waveform.plot(ch0_result["time"], y_data)
            self.ax_waveform.set_title("Waveform")
            self.ax_waveform.set_xlabel("Time (s)")
            self.ax_waveform.set_ylabel(y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_waveform.grid(True)
            self.canvas_waveform.draw()

        elif view_index == 1:
            # Waveform + Spectrum
            self.ax_top.clear()
            self.ax_top.plot(ch0_result["time"], y_data)
            self.ax_top.set_title("Waveform")
            self.ax_top.set_xlabel("Time (s)")
            self.ax_top.set_ylabel(y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_top.grid(True)
            self.canvas_top.draw()

            self.ax_bottom.clear()
            mask = fft_freqs <= 500
            self.freqs = fft_freqs[mask]
            self.fft_mags = fft_mags[mask]
            self.ax_bottom.plot(self.freqs, self.fft_mags)
            self.ax_bottom.set_title("Spectrum")
            self.ax_bottom.set_xlabel("Frequency (Hz)")
            self.ax_bottom.set_ylabel(f"{y_label} RMS")
            self.ax_bottom.grid(True)
            self.canvas_bottom.draw()

        elif view_index == 2:
            # Waveform (Ch1 + Ch2)
            self.ax_ch1.clear()
            self.ax_ch2.clear()

            self.ax_ch1.plot(ch0_result["time"], ch0_result["acceleration"], label="Ch0")
            self.ax_ch1.set_ylabel("Ch0 Acc (g)")
            self.ax_ch1.grid(True)

            self.ax_ch2.plot(ch1_result["time"], ch1_result["acceleration"], label="Ch1", color="orange")
            self.ax_ch2.set_ylabel("Ch1 Acc (g)")
            self.ax_ch2.set_xlabel("Time (s)")
            self.ax_ch2.grid(True)

            self.canvas_wave_ch.draw()

        elif view_index == 3:
            # Spectrum (Ch1 + Ch2)
            self.ax_ch1_spec.clear()
            self.ax_ch2_spec.clear()

            self.ax_ch1_spec.plot(ch0_result["fft_freqs"], ch0_result["fft_mags"], label="Ch0")
            self.ax_ch1_spec.set_ylabel("Ch0 FFT (g)")
            self.ax_ch1_spec.grid(True)

            self.ax_ch2_spec.plot(ch1_result["fft_freqs"], ch1_result["fft_mags"], label="Ch1", color="orange")
            self.ax_ch2_spec.set_ylabel("Ch1 FFT (g)")
            self.ax_ch2_spec.set_xlabel("Frequency (Hz)")
            self.ax_ch2_spec.grid(True)

            self.canvas_fft_ch.draw()

        # Update readings display (from Ch0 only)
        self.acc_input["input"].setText(f"{ch0_result['acc_peak']:.2f}")
        self.vel_input["input"].setText(f"{ch0_result['rms_fft']:.2f}")
        self.disp_input["input"].setText(f"{ch0_result['disp_pp']:.2f}")
        self.freq_input["input"].setText(f"{ch0_result['dom_freq']:.2f}")


    def on_waveform_click(self, event):
        if not hasattr(self, 't') or event.inaxes not in [self.ax_waveform, self.ax_top]:
            return
        x = event.xdata
        idx = min(range(len(self.t)), key=lambda i: abs(self.t[i] - x))
        t_val = self.t[idx]
        y_val = self.accel[idx]
        event.inaxes.annotate(f"({t_val:.3f}s, {y_val:.3f}g)", (t_val, y_val),
                              textcoords="offset points", xytext=(10, 10),
                              arrowprops=dict(arrowstyle="->", color='red'),
                              fontsize=9, color='red')
        event.canvas.draw()

    def on_fft_click(self, event):
        if not hasattr(self, 'freqs') or event.inaxes != self.ax_bottom:
            return
        x = event.xdata
        idx = min(range(len(self.freqs)), key=lambda i: abs(self.freqs[i] - x))
        f_val = self.freqs[idx]
        mag = self.fft_mags[idx]
        self.ax_bottom.annotate(f"({f_val:.1f} Hz, {mag:.3f})", (f_val, mag),
                                textcoords="offset points", xytext=(10, 10),
                                arrowprops=dict(arrowstyle="->", color='green'),
                                fontsize=9, color='green')
        self.canvas_bottom.draw()