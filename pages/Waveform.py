
from tkinter.tix import ComboBox
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QSizePolicy, QMenu, QStackedWidget, QComboBox
)
from PyQt6.QtWidgets import (QDialog,QTableWidget,QTableWidgetItem,QVBoxLayout,QPushButton)
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
        #self.daq = Mcc172Backend(board_num=0, channel=1,sensitivity=0.1, sample_rate=51200)
        #self.daq.setup()
        # self.daq.start_acquisition()
        # self.daq.auto_detect_channel()
        self.selected_quantity = "Acceleration"
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
        self.dual_waveform_view = self.build_dual_waveform_view()
        self.dual_spectrum_view = self.build_dual_spectrum_view()

        self.stacked_views.addWidget(self.default_view)
        self.stacked_views.addWidget(self.dual_view)
        self.stacked_views.addWidget(self.dual_waveform_view)
        self.stacked_views.addWidget(self.dual_spectrum_view)
        self.main_layout.addWidget(self.stacked_views)

        bottom_buttons = QHBoxLayout()
        for label in ["Traces", "Param", "Control", "Auto", "Cursor"]:
            if label == "Traces":
                self.traces_button = QPushButton(label)
                self.traces_menu = QMenu()
                self.traces_menu.addAction("Readings + Waveform", lambda: self.switch_trace_mode(0))
                self.traces_menu.addAction("Waveform + Spectrum", lambda: self.switch_trace_mode(1))
                self.traces_menu.addAction("waveform(ch1 + ch2)", lambda: self.switch_trace_mode(2))
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
                self.params_Menu.addAction("Input channels", lambda: self.input_channels())
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
    
    def input_channels(self):
        self.popup_input_channels = InputChannelsDialog(self)
        if self.popup_input_channels.exec():  # Waits until Apply or Close
            self.channel_configs = self.popup_input_channels.channel_config
            print("Received channel configuration in WaveformPage:")
            for config in self.channel_configs:
                print(config)

    def get_enabled_channels(self):
        enabled_channels = []
        if not hasattr(self, 'channel_configs'):
            return enabled_channels
        
        for config in self.channel_configs:
            if config["status"] == "ON":
                try:
                    ch_num = int(config["ch"])
                    board_num = (ch_num - 1) // 2
                    channel = (ch_num - 1) % 2
                    sensitivity_str = config["sensitivity"]
                    sensitivity = float(sensitivity_str.split(" ")[0])  # Extract numeric part
                    enabled_channels.append({
                        "board_num": board_num,
                        "channel": channel,
                        "sensitivity": sensitivity,
                    })
                except Exception as e:
                    print(f"Error processing channel config {config}: {e}")
        return enabled_channels
                



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
        self.fig_waveform_ch = Figure()
        self.canvas_waveform_ch = FigureCanvas(self.fig_waveform_ch)
        self.ax_ch1_waveform = self.fig_waveform_ch.add_subplot(211)
        self.ax_ch2_waveform = self.fig_waveform_ch.add_subplot(212)
        layout.addWidget(self.canvas_waveform_ch)
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
        enabled_channels = self.get_enabled_channels()
        if not enabled_channels:
            print("No enabled channels found. Please configure input channels.")
            return
        # self.daq = Mcc172Backend(board_num=enabled_channels[0]["board_num"],
        #                          channel=enabled_channels[0]["channel"],
        #                          sensitivity=enabled_channels[0]["sensitivity"],
        #                          sample_rate=51200)
        self.daq = Mcc172Backend(enabled_channels =enabled_channels, sample_rate=51200)
        self.daq.setup()
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

    def update_plot(self):
        self.t, self.accel, self.velocity, self.displacement, acc_peak, acc_rms, vel_rms, disp_pp, dom_freq,fft_freqs,fft_mags,freqs_vel,fft_mags_vel,fft_freqs_disp,fft_mags_disp,rms_fft= self.daq.get_latest_waveform()
        if len(self.t) == 0:
            return


        if self.selected_quantity == "Velocity":
            y_data = self.velocity
            N = len(y_data)
            y_label = "Velocity (mm/s)"
            pos_freqs = freqs_vel[:N // 2]
            mask = pos_freqs <= 500
            self.freqs = pos_freqs[mask]
            fft_mags = fft_mags_vel[mask]
            self.fft_mags = fft_mags


        elif self.selected_quantity == "Displacement":
            y_data = self.displacement
            y_label = "Displacement (μm)"
            N2 = len(y_data)
            pos_freqs = fft_freqs_disp[:N2 // 2]
            mask = pos_freqs <= 500 
            self.freqs = pos_freqs[mask]
            fft_mags = fft_mags_disp[mask]
            self.fft_mags = fft_mags
            
        else:
            y_data = self.accel
            N1 = len(y_data)
            y_label = "Acceleration (g)"
            pos_freqs = fft_freqs[:N1 // 2]
            mask = pos_freqs <=500
            self.freqs = pos_freqs[mask]
            fft_mags = fft_mags[mask]
            self.fft_mags = fft_mags

        view_index = self.stacked_views.currentIndex()
        if view_index == 0:
            self.ax_waveform.clear()
            self.ax_waveform.plot(self.t, y_data)
            self.ax_waveform.set_title("Waveform")
            self.ax_waveform.set_xlabel("Time (s)")
            self.ax_waveform.set_ylabel(y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_waveform.grid(True)
            self.canvas_waveform.draw()
        else:
            self.ax_top.clear()
            self.ax_top.plot(self.t, y_data)
            self.ax_top.set_title("Waveform")
            self.ax_top.set_xlabel("Time (s)")
            self.ax_top.set_ylabel(y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_top.grid(True)
            self.canvas_top.draw()

            self.ax_bottom.clear()
            #fft_result = np.fft.fft(y_data)
            
            #freqs = np.fft.fftfreq(N, 1 / self.daq.actual_rate)
            #fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
            
            self.ax_bottom.plot(self.freqs, self.fft_mags)
            self.ax_bottom.set_title("Spectrum")
            self.ax_bottom.set_xlabel("Frequency (Hz)")
            self.ax_bottom.set_ylabel(f"{y_label} RMS")
            self.ax_bottom.grid(True)
            self.canvas_bottom.draw()

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
class InputChannelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Channel and Sensor Configuration")
        self.setMinimumSize(600, 300)

        self.table = QTableWidget(6, 5)
        self.table.setHorizontalHeaderLabels(["Ch.", "Sensitivity", "Input Mode", "HP Fltr", "Status"])
        self.table.verticalHeader().setVisible(False)

        # Sample data
        default_data = [
            ["1", "0.1 g", "IEPE", "1Hz", "ON"],
            ["2", "0.1 g", "IEPE", "1Hz", "ON"],
            ["3", "0.1 g", "IEPE", "1Hz", "OFF"],
            ["4", "0.1 g", "IEPE", "1Hz", "ON"],
            ["5", "0.1 g", "IEPE", "1Hz", "OFF"],
            ["6", "0.1 g", "IEPE", "1Hz", "ON"]
        ]

        # Fill table
        for row, row_data in enumerate(default_data):
            for col, value in enumerate(row_data):
                if col == 4:  # Status column → dropdown
                    combo = QComboBox()
                    combo.addItems(["ON", "OFF"])
                    combo.setCurrentText(value)
                    self.table.setCellWidget(row, col, combo)
                else:
                    item = QTableWidgetItem(value)
                    if col == 0:
                        item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)  # Make Ch. read-only
                    self.table.setItem(row, col, item)

        layout = QVBoxLayout()
        layout.addWidget(self.table)

        # Apply Button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_and_close)
        layout.addWidget(apply_btn)

        self.setLayout(layout)

    def apply_and_close(self):
        self.channel_config = []
        for row in range(self.table.rowCount()):
            ch = self.table.item(row, 0).text()
            sensitivity = self.table.item(row, 1).text()
            input_mode = self.table.item(row, 2).text()
            hp_filter = self.table.item(row, 3).text()
            status_widget = self.table.cellWidget(row, 4)
            status = status_widget.currentText() if isinstance(status_widget, QComboBox) else "OFF"

            self.channel_config.append({
                "ch": int(ch),
                "sensitivity": sensitivity,
                "input_mode": input_mode,
                "hp_filter": hp_filter,
                "status": status
            })

        print("Collected Channel Configurations:")
        for config in self.channel_config:
            print(config)

        self.accept()  # Close dialog
from tkinter.tix import ComboBox
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QSizePolicy, QMenu, QStackedWidget, QComboBox
)
from PyQt6.QtWidgets import (QDialog,QTableWidget,QTableWidgetItem,QVBoxLayout,QPushButton)
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
        #self.daq = Mcc172Backend(board_num=0, channel=1,sensitivity=0.1, sample_rate=51200)
        #self.daq.setup()
        # self.daq.start_acquisition()
        # self.daq.auto_detect_channel()
        self.selected_quantity = "Acceleration"
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
        self.dual_waveform_view = self.build_dual_waveform_view()
        self.dual_spectrum_view = self.build_dual_spectrum_view()

        self.stacked_views.addWidget(self.default_view)
        self.stacked_views.addWidget(self.dual_view)
        self.stacked_views.addWidget(self.dual_waveform_view)
        self.stacked_views.addWidget(self.dual_spectrum_view)
        self.main_layout.addWidget(self.stacked_views)

        bottom_buttons = QHBoxLayout()
        for label in ["Traces", "Param", "Control", "Auto", "Cursor"]:
            if label == "Traces":
                self.traces_button = QPushButton(label)
                self.traces_menu = QMenu()
                self.traces_menu.addAction("Readings + Waveform", lambda: self.switch_trace_mode(0))
                self.traces_menu.addAction("Waveform + Spectrum", lambda: self.switch_trace_mode(1))
                self.traces_menu.addAction("waveform(ch1 + ch2)", lambda: self.switch_trace_mode(2))
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
                self.params_Menu.addAction("Input channels", lambda: self.input_channels())
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
    
    def input_channels(self):
        self.popup_input_channels = InputChannelsDialog(self)
        if self.popup_input_channels.exec():  # Waits until Apply or Close
            self.channel_configs = self.popup_input_channels.channel_config
            print("Received channel configuration in WaveformPage:")
            for config in self.channel_configs:
                print(config)

    def get_enabled_channels(self):
        enabled_channels = []
        if not hasattr(self, 'channel_configs'):
            return enabled_channels
        
        for config in self.channel_configs:
            if config["status"] == "ON":
                try:
                    ch_num = int(config["ch"])
                    board_num = (ch_num - 1) // 2
                    channel = (ch_num - 1) % 2
                    sensitivity_str = config["sensitivity"]
                    sensitivity = float(sensitivity_str.split(" ")[0])  # Extract numeric part
                    enabled_channels.append({
                        "board_num": board_num,
                        "channel": channel,
                        "sensitivity": sensitivity,
                    })
                except Exception as e:
                    print(f"Error processing channel config {config}: {e}")
        return enabled_channels
                



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
        self.fig_waveform_ch = Figure()
        self.canvas_waveform_ch = FigureCanvas(self.fig_waveform_ch)
        self.ax_ch1_waveform = self.fig_waveform_ch.add_subplot(211)
        self.ax_ch2_waveform = self.fig_waveform_ch.add_subplot(212)
        layout.addWidget(self.canvas_waveform_ch)
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
        enabled_channels = self.get_enabled_channels()
        if not enabled_channels:
            print("No enabled channels found. Please configure input channels.")
            return
        # self.daq = Mcc172Backend(board_num=enabled_channels[0]["board_num"],
        #                          channel=enabled_channels[0]["channel"],
        #                          sensitivity=enabled_channels[0]["sensitivity"],
        #                          sample_rate=51200)
        self.daq = Mcc172Backend(enabled_channels =enabled_channels, sample_rate=51200)
        self.daq.setup()
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

    def update_plot(self):
        self.t, self.accel, self.velocity, self.displacement, acc_peak, acc_rms, vel_rms, disp_pp, dom_freq,fft_freqs,fft_mags,freqs_vel,fft_mags_vel,fft_freqs_disp,fft_mags_disp,rms_fft= self.daq.get_latest_waveform()
        if len(self.t) == 0:
            return


        if self.selected_quantity == "Velocity":
            y_data = self.velocity
            N = len(y_data)
            y_label = "Velocity (mm/s)"
            pos_freqs = freqs_vel[:N // 2]
            mask = pos_freqs <= 500
            self.freqs = pos_freqs[mask]
            fft_mags = fft_mags_vel[mask]
            self.fft_mags = fft_mags


        elif self.selected_quantity == "Displacement":
            y_data = self.displacement
            y_label = "Displacement (μm)"
            N2 = len(y_data)
            pos_freqs = fft_freqs_disp[:N2 // 2]
            mask = pos_freqs <= 500 
            self.freqs = pos_freqs[mask]
            fft_mags = fft_mags_disp[mask]
            self.fft_mags = fft_mags
            
        else:
            y_data = self.accel
            N1 = len(y_data)
            y_label = "Acceleration (g)"
            pos_freqs = fft_freqs[:N1 // 2]
            mask = pos_freqs <=500
            self.freqs = pos_freqs[mask]
            fft_mags = fft_mags[mask]
            self.fft_mags = fft_mags

        view_index = self.stacked_views.currentIndex()
        if view_index == 0:
            self.ax_waveform.clear()
            self.ax_waveform.plot(self.t, y_data)
            self.ax_waveform.set_title("Waveform")
            self.ax_waveform.set_xlabel("Time (s)")
            self.ax_waveform.set_ylabel(y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_waveform.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_waveform.grid(True)
            self.canvas_waveform.draw()
        else:
            self.ax_top.clear()
            self.ax_top.plot(self.t, y_data)
            self.ax_top.set_title("Waveform")
            self.ax_top.set_xlabel("Time (s)")
            self.ax_top.set_ylabel(y_label)
            margin = (max(y_data) - min(y_data)) * 0.1 or 0.2
            self.ax_top.set_ylim(min(y_data) - margin, max(y_data) + margin)
            self.ax_top.grid(True)
            self.canvas_top.draw()

            self.ax_bottom.clear()
            #fft_result = np.fft.fft(y_data)
            
            #freqs = np.fft.fftfreq(N, 1 / self.daq.actual_rate)
            #fft_mags = (2.0 / N) * np.abs(fft_result[:N // 2])
            
            self.ax_bottom.plot(self.freqs, self.fft_mags)
            self.ax_bottom.set_title("Spectrum")
            self.ax_bottom.set_xlabel("Frequency (Hz)")
            self.ax_bottom.set_ylabel(f"{y_label} RMS")
            self.ax_bottom.grid(True)
            self.canvas_bottom.draw()

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
class InputChannelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Channel and Sensor Configuration")
        self.setMinimumSize(600, 300)

        self.table = QTableWidget(6, 5)
        self.table.setHorizontalHeaderLabels(["Ch.", "Sensitivity", "Input Mode", "HP Fltr", "Status"])
        self.table.verticalHeader().setVisible(False)

        # Sample data
        default_data = [
            ["1", "0.1 g", "IEPE", "1Hz", "ON"],
            ["2", "0.1 g", "IEPE", "1Hz", "ON"],
            ["3", "0.1 g", "IEPE", "1Hz", "OFF"],
            ["4", "0.1 g", "IEPE", "1Hz", "ON"],
            ["5", "0.1 g", "IEPE", "1Hz", "OFF"],
            ["6", "0.1 g", "IEPE", "1Hz", "ON"]
        ]

        # Fill table
        for row, row_data in enumerate(default_data):
            for col, value in enumerate(row_data):
                if col == 4:  # Status column → dropdown
                    combo = QComboBox()
                    combo.addItems(["ON", "OFF"])
                    combo.setCurrentText(value)
                    self.table.setCellWidget(row, col, combo)
                else:
                    item = QTableWidgetItem(value)
                    if col == 0:
                        item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)  # Make Ch. read-only
                    self.table.setItem(row, col, item)

        layout = QVBoxLayout()
        layout.addWidget(self.table)

        # Apply Button
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_and_close)
        layout.addWidget(apply_btn)

        self.setLayout(layout)

    def apply_and_close(self):
        self.channel_config = []
        for row in range(self.table.rowCount()):
            ch = self.table.item(row, 0).text()
            sensitivity = self.table.item(row, 1).text()
            input_mode = self.table.item(row, 2).text()
            hp_filter = self.table.item(row, 3).text()
            status_widget = self.table.cellWidget(row, 4)
            status = status_widget.currentText() if isinstance(status_widget, QComboBox) else "OFF"

            self.channel_config.append({
                "ch": int(ch),
                "sensitivity": sensitivity,
                "input_mode": input_mode,
                "hp_filter": hp_filter,
                "status": status
            })

        print("Collected Channel Configurations:")
        for config in self.channel_config:
            print(config)

        self.accept()  # Close dialog