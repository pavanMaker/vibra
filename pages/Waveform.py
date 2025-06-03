from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QSizePolicy, QMenu, QStackedWidget, QComboBox
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
        self.daq = Mcc172Backend(sample_rate=51200)
        self.daq.setup()
        # self.daq.start_acquisition()
        # self.daq.auto_detect_channel()
        self.selected_quantity = "Acceleration"
        #self.is_running = False  # Measurement state tracker

        self.setup_ui()
        self.selected_trace_mode = "readings_waveform"
        self.top_selected_channel = 0
        self.bottom_selected_channel = 1
        
        self.start_clock()
          # Default trace mode
        

        self.timer = QTimer()

        self.timer.timeout.connect(self.update_plot)

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Top Navbar
        nav_bar = QHBoxLayout()
        title_label = QLabel("Waveform & Spectrum")
        title_label.setStyleSheet("font-size: 1.5rem; font-weight: bold;")
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-weight: bold; font-size: 1rem;")

        nav_bar.addWidget(title_label)
        nav_bar.addWidget(self.time_label)
        self.main_layout.addLayout(nav_bar)

        # Trace + Window dropdown logic (Step 1)
        self.traces_button = QPushButton("Traces")
        self.traces_menu = QMenu()

        trace_modes = {
            "Readings + Waveform": "readings_waveform",
            "Waveform + Spectrum": "waveform_spectrum",
            "Waveform + Waveform": "waveform_waveform",
            "Spectrum + Spectrum": "spectrum_spectrum",
            "Readings + Readings": "readings_readings",
            "Readings + Spectrum": "readings_spectrum"
        }

        for label, mode in trace_modes.items():
            self.traces_menu.addAction(label, lambda checked=False, m=mode: self.switch_trace_mode(m))

        self.traces_button.setMenu(self.traces_menu)

        trace_layout = QHBoxLayout()
        trace_layout.addWidget(QLabel("Trace + Window:"))
        trace_layout.addWidget(self.traces_button)
        trace_layout.addStretch()
        self.main_layout.addLayout(trace_layout)

        # Step 2: Channel selectors (Top & Bottom)
        self.top_channel_combo = QComboBox()
        self.top_channel_combo.addItems(["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6"])
        self.top_channel_combo.currentIndexChanged.connect(self.on_channel_change)

        self.bottom_channel_combo = QComboBox()
        self.bottom_channel_combo.addItems(["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6"])
        self.bottom_channel_combo.currentIndexChanged.connect(self.on_channel_change)

        channel_select_layout = QHBoxLayout()
        channel_select_layout.addWidget(QLabel("Top Channel:"))
        channel_select_layout.addWidget(self.top_channel_combo)
        channel_select_layout.addSpacing(50)
        channel_select_layout.addWidget(QLabel("Bottom Channel:"))
        channel_select_layout.addWidget(self.bottom_channel_combo)
        self.main_layout.addLayout(channel_select_layout)

        # Views Area (stacked)
        self.stacked_views = QStackedWidget()
        self.default_view = self.build_readings_waveform_view()
        self.dual_view = self.build_waveform_spectrum_view()

        self.stacked_views.addWidget(self.default_view)
        self.stacked_views.addWidget(self.dual_view)
        self.main_layout.addWidget(self.stacked_views)

        # Bottom Buttons
        bottom_buttons = QHBoxLayout()
        for label in ["Param", "Control", "Auto", "Cursor"]:
            if label == "Param":
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

        # Start/Stop buttons
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

    def on_channel_change(self):
        self.top_selected_channel = self.top_channel_combo.currentIndex()
        self.bottom_selected_channel = self.bottom_channel_combo.currentIndex()
    
    def get_selected_channel_configs(self):
        configs = []

        mode = self.selected_trace_mode

        if mode in ["readings_waveform", "waveform_spectrum", "readings_readings", "readings_spectrum"]:
            # Only top channel needed
            configs.append({
                "board_num": self.top_selected_channel // 2,
                "channel": self.top_selected_channel % 2,
                "sensitivity": 0.1
            })
        elif mode in ["waveform_waveform", "spectrum_spectrum"]:
            # Both top and bottom channels needed
            if self.top_selected_channel != self.bottom_selected_channel:
                configs.append({
                    "board_num": self.top_selected_channel // 2,
                    "channel": self.top_selected_channel % 2,
                    "sensitivity": 0.1
                })
                configs.append({
                    "board_num": self.bottom_selected_channel // 2,
                    "channel": self.bottom_selected_channel % 2,
                    "sensitivity": 0.1
                })
            else:
                # Same channel selected for both top and bottom (edge case)
                configs.append({
                    "board_num": self.top_selected_channel // 2,
                    "channel": self.top_selected_channel % 2,
                    "sensitivity": 0.1
                })
        else:
            print(f"⚠️ Unrecognized trace mode '{mode}', defaulting to top channel only.")
            configs.append({
                "board_num": self.top_selected_channel // 2,
                "channel": self.top_selected_channel % 2,
                "sensitivity": 0.1
            })
        print("configured channels:",configs)

        return configs
    


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
        pass
    def build_dual_spectrum_view(self):
        pass
    def build_dual_readings_view(self):
        pass
    def trace_window_settings(self):
        pass


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

    def switch_trace_mode(self, mode):
        self.selected_trace_mode = mode
        print(f"Selected Trace Mode: {mode}")

        if mode == "readings_waveform":
            self.stacked_views.setCurrentIndex(0)  # Top: Readings, Bottom: Waveform

        elif mode == "waveform_spectrum":
            self.stacked_views.setCurrentIndex(1)  # Top: Waveform, Bottom: Spectrum

        elif mode in ["waveform_waveform", "spectrum_spectrum", "readings_readings", "readings_spectrum"]:
            dual_view = QWidget()
            layout = QVBoxLayout(dual_view)

            self.fig_top = Figure(figsize=(6, 3))
            self.canvas_top = FigureCanvas(self.fig_top)
            self.ax_top = self.fig_top.add_subplot(111)
            layout.addWidget(self.canvas_top)

            self.fig_bottom = Figure(figsize=(6, 3))
            self.canvas_bottom = FigureCanvas(self.fig_bottom)
            self.ax_bottom = self.fig_bottom.add_subplot(111)
            layout.addWidget(self.canvas_bottom)

            self.stacked_views.addWidget(dual_view)
            self.stacked_views.setCurrentWidget(dual_view)

        else:
            print(f"🧩 Trace mode '{mode}' is not implemented.")



    def set_active_channels(self, configs):
        self.active_configs = configs
    


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
    # Step 1: Get selected channels
        selected_configs = self.get_selected_channel_configs()

        # Step 2: Update backend
        self.daq.set_active_channels(selected_configs)  # This method should be in your backend

        # Step 3: Setup and start
        self.daq.setup()
        self.daq.start_acquisition()
        self.timer.start(1000)

        # Step 4: Update UI state
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)

        print("📡 Measurement started with channels:")
        for cfg in selected_configs:
            print(f"   → Board {cfg['board_num']} Channel {cfg['channel']}")



    def stop_measurement(self):
        self.daq.stop_scan()
        self.timer.stop()
        #self.is_running = False
        self.start_button.setVisible(True)
        self.stop_button.setVisible(False)
        print("Measurement stopped.")

    def update_plot(self):
        results = self.daq.get_latest_waveform()

        if not results or len(results) == 0:
            print("⚠️ No data received.")
            return

        mode = self.selected_trace_mode
        num_results = len(results)

        if mode in ["waveform_waveform", "spectrum_spectrum", "readings_readings"]:
            top_result = results[0]
            bottom_result = results[1] if num_results == 2 else results[0]
        else:
            top_result = results[0]
            bottom_result = None

        # Extract readings
        acc_peak = top_result.get("acc_peak", 0)
        acc_rms = top_result.get("acc_rms", 0)
        vel_rms = top_result.get("vel_rms", 0)
        disp_pp = top_result.get("disp_pp", 0)
        dom_freq = top_result.get("dom_freq", 0)
        rms_fft = top_result.get("rms_fft", 0)

        def get_y_and_label(result):
            if self.selected_quantity == "Velocity":
                return result["velocity"], "Velocity (mm/s)", result["fft_freqs_vel"], result["fft_mags_vel"]
            elif self.selected_quantity == "Displacement":
                return result["displacement"], "Displacement (μm)", result["fft_freqs_disp"], result["fft_mags_disp"]
            else:
                return result["accel"], "Acceleration (g)", result["fft_freqs"], result["fft_mags"]

        if mode == "readings_waveform":
            y_data, y_label, _, _ = get_y_and_label(top_result)
            t = top_result["t"]
            self.ax_waveform.clear()
            self.ax_waveform.plot(t, y_data)
            self.ax_waveform.set(title="Waveform", xlabel="Time (s)", ylabel=y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_waveform.grid(True)
            self.canvas_waveform.draw()

        elif mode == "waveform_spectrum":
            y_data, y_label, _, _ = get_y_and_label(top_result)
            t = top_result["t"]
            self.ax_top.clear()
            self.ax_top.plot(t, y_data)
            self.ax_top.set(title="Waveform", xlabel="Time (s)", ylabel=y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_top.grid(True)
            self.canvas_top.draw()

            _, y_label_b, fft_freqs, fft_mags = get_y_and_label(bottom_result or top_result)
            self.ax_bottom.clear()
            self.ax_bottom.plot(fft_freqs, fft_mags)
            self.ax_bottom.set(title="Spectrum", xlabel="Frequency (Hz)", ylabel=f"{y_label_b} RMS")
            self.ax_bottom.grid(True)
            self.canvas_bottom.draw()

        elif mode == "waveform_waveform":
            y1, label1, _, _ = get_y_and_label(top_result)
            y2, label2, _, _ = get_y_and_label(bottom_result or top_result)
            t1 = top_result["t"]
            t2 = bottom_result["t"] if bottom_result else t1

            self.ax_top.clear()
            self.ax_top.plot(t1, y1)
            self.ax_top.set(title="Waveform - Top", ylabel=label1)
            self.ax_top.grid(True)

            self.ax_bottom.clear()
            self.ax_bottom.plot(t2, y2)
            self.ax_bottom.set(title="Waveform - Bottom", ylabel=label2)
            self.ax_bottom.grid(True)

            self.canvas_top.draw()
            self.canvas_bottom.draw()

        elif mode == "spectrum_spectrum":
            f1 = top_result["fft_freqs"]
            m1 = top_result["fft_mags"]
            f2 = bottom_result["fft_freqs"] if bottom_result else f1
            m2 = bottom_result["fft_mags"] if bottom_result else m1

            self.ax_top.clear()
            self.ax_top.plot(f1, m1)
            self.ax_top.set(title="Spectrum - Top", ylabel="Magnitude")
            self.ax_top.grid(True)

            self.ax_bottom.clear()
            self.ax_bottom.plot(f2, m2)
            self.ax_bottom.set(title="Spectrum - Bottom", ylabel="Magnitude")
            self.ax_bottom.grid(True)

            self.canvas_top.draw()
            self.canvas_bottom.draw()

        elif mode == "readings_readings":
            self.acc_input["input"].setText(f"{top_result['acc_peak']:.2f}")
            self.vel_input["input"].setText(f"{top_result['vel_rms']:.2f}")
            self.disp_input["input"].setText(f"{top_result['disp_pp']:.2f}")
            self.freq_input["input"].setText(f"{top_result['dom_freq']:.2f}")
            print(f"CH2 → Acc: {bottom_result['acc_peak']:.2f}, Vel: {bottom_result['vel_rms']:.2f}, "
                f"Disp: {bottom_result['disp_pp']:.2f}, Freq: {bottom_result['dom_freq']:.2f}")

        elif mode == "readings_spectrum":
            self.acc_input["input"].setText(f"{top_result['acc_peak']:.2f}")
            self.vel_input["input"].setText(f"{top_result['vel_rms']:.2f}")
            self.disp_input["input"].setText(f"{top_result['disp_pp']:.2f}")
            self.freq_input["input"].setText(f"{top_result['dom_freq']:.2f}")

            f2 = bottom_result["fft_freqs"] if bottom_result else top_result["fft_freqs"]
            m2 = bottom_result["fft_mags"] if bottom_result else top_result["fft_mags"]

            self.ax_bottom.clear()
            self.ax_bottom.plot(f2, m2)
            self.ax_bottom.set(title="Spectrum (Bottom)")
            self.ax_bottom.grid(True)
            self.canvas_bottom.draw()

        else:
            print(f"🧩 Trace mode '{mode}' not yet implemented in update_plot()")

        # Always update top reading boxes with top_result
        self.acc_input["input"].setText(f"{acc_peak:.2f}")
        self.vel_input["input"].setText(f"{rms_fft:.2f}")
        self.disp_input["input"].setText(f"{disp_pp:.2f}")
        self.freq_input["input"].setText(f"{dom_freq:.2f}")



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