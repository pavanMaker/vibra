# WaveformPage.py (final version with trace + window + dual-channel + full compatibility)

# ... keep your imports here ...
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QSizePolicy, QMenu, QStackedWidget, QComboBox, QDialog, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from backend.mcc_backend import Mcc172Backend
from pages.AnalysisParameters import AnalysisParameter


class WaveformPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.selected_quantity = "Acceleration"

        self.ax_waveform = None
        self.ax_top = None
        self.ax_bottom = None
        self.ax_ch1_waveform = None
        self.ax_ch2_waveform = None
        self.ax_ch1_spec = None
        self.ax_ch2_spec = None
        self.ax_spectrum = None

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

        # Trace views
        self.stacked_views = QStackedWidget()
        self.default_view = self.build_readings_waveform_view()
        self.dual_view = self.build_waveform_spectrum_view()
        self.dual_waveform_view = self.build_dual_waveform_view()
        self.dual_spectrum_view = self.build_dual_spectrum_view()
        self.readings_spectrum_view = self.build_readings_spectrum_view()
        self.dual_readings_view = self.build_dual_readings_view()

        self.stacked_views.addWidget(self.default_view)        # 0
        self.stacked_views.addWidget(self.dual_view)           # 1
        self.stacked_views.addWidget(self.dual_waveform_view)  # 2
        self.stacked_views.addWidget(self.dual_spectrum_view)  # 3
        self.stacked_views.addWidget(self.readings_spectrum_view)  # 4
        self.stacked_views.addWidget(self.dual_readings_view)      # 5
        self.main_layout.addWidget(self.stacked_views)

        # Trace Navigation + Channel Select
        self.trace_window_settings()

        # Bottom controls
        bottom_buttons = QHBoxLayout()
        for label in ["Traces", "Param", "Control", "Auto", "Cursor"]:
            btn = QPushButton(label)
            btn.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold;")
            if label == "Traces":
                self.traces_button = btn
                self.traces_menu = QMenu()
                for i, txt in enumerate([
                    "Readings + Waveform", "Waveform + Spectrum", "Waveform (ch1 + ch2)",
                    "Spectrum (ch1 + ch2)", "Readings + Spectrum", "Dual Readings"
                ]):
                    self.traces_menu.addAction(txt, lambda idx=i: self.switch_trace_mode(idx))
                self.traces_button.setMenu(self.traces_menu)
            elif label == "Param":
                self.params_button = btn
                self.params_Menu = QMenu(btn)
                self.params_Menu.addAction("Analysis Parameters", self.analysis_parameter)
                self.params_Menu.addAction("Input channels", self.input_channels)
                btn.setMenu(self.params_Menu)
            bottom_buttons.addWidget(btn)

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

    def trace_window_settings(self):
        nav_layout = QHBoxLayout()
        self.prev_trace_btn = QPushButton("←")
        self.next_trace_btn = QPushButton("→")
        self.prev_trace_btn.clicked.connect(self.prev_trace)
        self.next_trace_btn.clicked.connect(self.next_trace)
        self.trace_combo = QComboBox()
        self.trace_combo.addItems([
            "Readings + Waveform", "Waveform + Spectrum", "Waveform (ch1 + ch2)",
            "Spectrum (ch1 + ch2)", "Readings + Spectrum", "Dual Readings"
        ])
        self.trace_combo.currentIndexChanged.connect(self.switch_trace_mode)
        nav_layout.addWidget(self.prev_trace_btn)
        nav_layout.addWidget(QLabel("Select Trace View:"))
        nav_layout.addWidget(self.trace_combo)
        nav_layout.addWidget(self.next_trace_btn)

        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("Top View Channel:"))
        self.top_channel_combo = QComboBox()
        self.top_channel_combo.addItems([f"CH{i+1}" for i in range(6)])
        self.top_channel_combo.currentIndexChanged.connect(self.update_top_channel)
        ch_layout.addWidget(self.top_channel_combo)

        ch_layout.addSpacing(20)
        ch_layout.addWidget(QLabel("Bottom View Channel:"))
        self.bottom_channel_combo = QComboBox()
        self.bottom_channel_combo.addItems([f"CH{i+1}" for i in range(6)])
        self.bottom_channel_combo.currentIndexChanged.connect(self.update_bottom_channel)
        ch_layout.addWidget(self.bottom_channel_combo)

        self.main_layout.insertLayout(1, nav_layout)
        self.main_layout.insertLayout(2, ch_layout)

        self.current_trace_index = 0
        self.current_top_channel = 0
        self.current_bottom_channel = 1
        self.top_channel_combo.currentIndexChanged.connect(self.restart_measurement)
        self.bottom_channel_combo.currentIndexChanged.connect(self.restart_measurement)

    def restart_measurement(self):
        self.stop_measurement()
        self.start_measurement()


    def prev_trace(self): self.trace_combo.setCurrentIndex((self.trace_combo.currentIndex() - 1) % 6)
    def next_trace(self): self.trace_combo.setCurrentIndex((self.trace_combo.currentIndex() + 1) % 6)
    def switch_trace_mode(self, index): self.stacked_views.setCurrentIndex(index)

    def update_top_channel(self, idx): self.current_top_channel = idx
    def update_bottom_channel(self, idx): self.current_bottom_channel = idx

    def start_clock(self):
        self.clock = QTimer()
        self.clock.timeout.connect(self.update_time)
        self.clock.start(1000)
        self.update_time()

    def update_time(self):
        now = QDateTime.currentDateTime()
        self.time_label.setText(now.toString("HH:mm:ss"))

    def analysis_parameter(self):
        self.popup = AnalysisParameter(self)
        self.popup.show()

    def input_channels(self):
        self.popup_input_channels = InputChannelsDialog(self)
        if self.popup_input_channels.exec():
            self.channel_configs = self.popup_input_channels.channel_config
            print("Received channel configuration in WaveformPage:")
            for config in self.channel_configs:
                print(config)

    def get_enabled_channels(self):
            selected = set([self.current_top_channel, self.current_bottom_channel])
            configs = []

            if not hasattr(self, 'channel_configs'):
                return configs

            for ch in selected:
                for cfg in self.channel_configs:
                    if cfg["ch"] == ch + 1:  # ch is 0-indexed, cfg["ch"] is 1-indexed
                        board = (ch) // 2
                        local_ch = (ch) % 2
                        sens = float(cfg["sensitivity"].split(" ")[0])
                        configs.append({
                            "board_num": board,
                            "channel": local_ch,
                            "sensitivity": sens
                        })
            return configs






    # def get_enabled_channels(self):
    #     enabled_channels = []
    #     if not hasattr(self, 'channel_configs'): return enabled_channels
    #     for config in self.channel_configs:
    #         try:
    #             ch = int(config["ch"])
    #             board = (ch - 1) // 2
    #             local_ch = (ch - 1) % 2
    #             sens = float(config["sensitivity"].split(" ")[0])
    #             enabled_channels.append({"board_num": board, "channel": local_ch, "sensitivity": sens})
    #         except: pass
    #     return enabled_channels

    def start_measurement(self):
        enabled_channels = self.get_enabled_channels()
        if not enabled_channels:
            print("No enabled channels found.")
            return
        selected_indices = [cfg['board_num'] * 2 + cfg['channel'] for cfg in enabled_channels]
        self.daq = Mcc172Backend(channel_configs=enabled_channels, selected_channels=selected_indices, sample_rate=51200)
        self.daq.setup()
        self.daq.start_acquisition()
        self.timer.start(1000)
        self.start_button.setVisible(False)
        self.stop_button.setVisible(True)

    def stop_measurement(self):
        self.daq.stop_scan()
        self.timer.stop()
        self.start_button.setVisible(True)
        self.stop_button.setVisible(False)
        print("Measurement stopped.")

    
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
        self.figure_waveform = Figure()
        self.canvas_waveform = FigureCanvas(self.figure_waveform)
        self.ax_waveform = self.figure_waveform.add_subplot(111)
        layout.addWidget(self.canvas_waveform)
        return widget

    def build_waveform_spectrum_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.figure_top = Figure()
        self.canvas_top = FigureCanvas(self.figure_top)
        self.ax_top = self.figure_top.add_subplot(111)
        layout.addWidget(self.canvas_top)
        self.figure_bottom = Figure()
        self.canvas_bottom = FigureCanvas(self.figure_bottom)
        self.ax_bottom = self.figure_bottom.add_subplot(111)
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

    def build_readings_spectrum_view(self):
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
        self.figure_spectrum = Figure()
        self.canvas_spectrum = FigureCanvas(self.figure_spectrum)
        self.ax_spectrum = self.figure_spectrum.add_subplot(111)
        layout.addWidget(self.canvas_spectrum)
        return widget
    
    def build_dual_readings_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # CH1 readings
        ch1_layout = QHBoxLayout()
        self.ch1_acc = self.create_reading_box("Ch1 Acc:", "g (peak)")
        self.ch1_vel = self.create_reading_box("Ch1 Vel:", "mm/s (RMS)")
        self.ch1_disp = self.create_reading_box("Ch1 Disp:", "µm (P-P)")
        self.ch1_freq = self.create_reading_box("Ch1 Freq:", "Hz")
        for box in [self.ch1_acc, self.ch1_vel, self.ch1_disp, self.ch1_freq]:
            ch1_layout.addLayout(box["layout"])
        layout.addLayout(ch1_layout)

        # CH2 readings
        ch2_layout = QHBoxLayout()
        self.ch2_acc = self.create_reading_box("Ch2 Acc:", "g (peak)")
        self.ch2_vel = self.create_reading_box("Ch2 Vel:", "mm/s (RMS)")
        self.ch2_disp = self.create_reading_box("Ch2 Disp:", "µm (P-P)")
        self.ch2_freq = self.create_reading_box("Ch2 Freq:", "Hz")
        for box in [self.ch2_acc, self.ch2_vel, self.ch2_disp, self.ch2_freq]:
            ch2_layout.addLayout(box["layout"])
        layout.addLayout(ch2_layout)

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
   
    def update_plot(self):
        results = self.daq.get_latest_waveform()
        print(f"Results received: {len(results)} channels")
        if not results or len(results) < 2: return

        top_idx = self.current_top_channel
        bottom_idx = self.current_bottom_channel
        if max(top_idx, bottom_idx) >= len(results): return

        top = results[top_idx]
        bottom = results[bottom_idx]

        def extract(res): return res[0], res[1], res[2], res[3], res[9], res[10], res[11], res[12], res[13], res[14]
        t1, a1, v1, d1, fx1, fa1, fxv1, fv1, fxd1, fd1 = extract(top)
        t2, a2, v2, d2, fx2, fa2, fxv2, fv2, fxd2, fd2 = extract(bottom)

        if self.selected_quantity == "Acceleration":
            y1, y2, fft1, fft2, f1, f2 = a1, a2, fa1, fa2, fx1, fx2
        elif self.selected_quantity == "Velocity":
            y1, y2, fft1, fft2, f1, f2 = v1, v2, fv1, fv2, fxv1, fxv2
        elif self.selected_quantity == "Displacement":
            y1, y2, fft1, fft2, f1, f2 = d1, d2, fd1, fd2, fxd1, fxd2
        else:
            return

        idx = self.trace_combo.currentIndex()
        if idx == 0 and self.ax_waveform:
            self.ax_waveform.clear()
            self.ax_waveform.plot(t1, y1)
            self.canvas_waveform.draw()
        elif idx == 1 and self.ax_top and self.ax_bottom:
            self.ax_top.clear(); self.ax_top.plot(t1, y1)
            self.ax_bottom.clear(); self.ax_bottom.plot(f1, fft1)
            self.canvas_top.draw(); self.canvas_bottom.draw()
        elif idx == 2 and self.ax_ch1_waveform and self.ax_ch2_waveform:
            self.ax_ch1_waveform.clear(); self.ax_ch1_waveform.plot(t1, y1)
            self.ax_ch2_waveform.clear(); self.ax_ch2_waveform.plot(t2, y2)
            self.canvas_waveform_ch.draw()
        elif idx == 3 and self.ax_ch1_spec and self.ax_ch2_spec:
            self.ax_ch1_spec.clear(); self.ax_ch1_spec.plot(f1, fft1)
            self.ax_ch2_spec.clear(); self.ax_ch2_spec.plot(f2, fft2)
            self.canvas_fft_ch.draw()
        elif idx == 4 and self.ax_spectrum:
            self.ax_spectrum.clear()
            self.ax_spectrum.plot(f1, fft1)
            self.canvas_spectrum.draw()
        elif idx == 5:  # Dual Readings View
            self.ch1_acc["input"].setText(f"{top[4]:.2f}")
            self.ch1_vel["input"].setText(f"{top[6]:.2f}")
            self.ch1_disp["input"].setText(f"{top[7]:.2f}")
            self.ch1_freq["input"].setText(f"{top[8]:.2f}")

            self.ch2_acc["input"].setText(f"{bottom[4]:.2f}")
            self.ch2_vel["input"].setText(f"{bottom[6]:.2f}")
            self.ch2_disp["input"].setText(f"{bottom[7]:.2f}")
            self.ch2_freq["input"].setText(f"{bottom[8]:.2f()}")


class InputChannelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Channel and Sensor Configuration")
        self.setMinimumSize(600, 300)

        self.table = QTableWidget(6, 5)
        self.table.setHorizontalHeaderLabels(["Ch.", "Sensitivity", "Input Mode", "HP Fltr"])
        self.table.verticalHeader().setVisible(False)

        # Sample data
        default_data = [
            ["1", "0.1 g", "IEPE", "1Hz"],
            ["2", "0.1 g", "IEPE", "1Hz"],
            ["3", "0.1 g", "IEPE", "1Hz"],
            ["4", "0.1 g", "IEPE", "1Hz"],
            ["5", "0.1 g", "IEPE", "1Hz"],
            ["6", "0.1 g", "IEPE", "1Hz"]
        ]

        # Fill table
        # for row, row_data in enumerate(default_data):
        #     for col, value in enumerate(row_data):
        #         if col == 4:  # Status column → dropdown
        #             combo = QComboBox()
        #             combo.addItems(["ON", "OFF"])
        #             combo.setCurrentText(value)
        #             self.table.setCellWidget(row, col, combo)
        #         else:
        #             item = QTableWidgetItem(value)
        #             if col == 0:
        #                 item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)  # Make Ch. read-only
        #             self.table.setItem(row, col, item)

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
            
            self.channel_config.append({
                "ch": int(ch),
                "sensitivity": sensitivity,
                "input_mode": input_mode,
                "hp_filter": hp_filter,
                
            })

        print("Collected Channel Configurations:")
        for config in self.channel_config:
            print(config)

        self.accept()  # Close dialog

