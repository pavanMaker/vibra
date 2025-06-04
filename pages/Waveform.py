
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
        self.daq = Mcc172Backend(board_num=0, channel=[0,1], sensitivity=0.1, sample_rate=51200)
        self.daq.setup()
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
        self.dual_waveform_view = self.build_waveform_waveform_view()
        self.dual_spectrum_view =  self.build_spectrum_spectrum_view()
        self.readings_view  = self.build_reading_reading_view()

        self.stacked_views.addWidget(self.default_view)
        self.stacked_views.addWidget(self.dual_view)
        self.stacked_views.addWidget(self.dual_waveform_view)
        self.stacked_views.addWidget(self.dual_spectrum_view)
        self.stacked_views.addWidget(self.readings_view)
        self.main_layout.addWidget(self.stacked_views)

        bottom_buttons = QHBoxLayout()
        for label in ["Traces", "Param", "Control", "Auto", "Cursor"]:
            if label == "Traces":
                self.traces_button = QPushButton(label)
                self.traces_menu = QMenu()
                self.traces_menu.addAction("Readings + Waveform", lambda: self.switch_trace_mode(0))
                self.traces_menu.addAction("Waveform + Spectrum", lambda: self.switch_trace_mode(1))
                self.traces_menu.addAction("Waveform + Waveform", lambda: self.switch_trace_mode(2))
                self.traces_menu.addAction("Spectrum + spectrum", lambda: self.switch_trace_mode(3))
                self.traces_menu.addAction("Readings + Readings", lambda: self.switch_trace_mode(4))
                self.traces_menu.addAction
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
    
    def build_waveform_waveform_view(self):
        widget = QWidget()
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
        self.daq = Mcc172Backend(board_num=0, channel=[0,1], sensitivity=0.1, sample_rate=51200)
        self.daq.setup()
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
        self.dual_waveform_view = self.build_waveform_waveform_view()
        self.dual_spectrum_view =  self.build_spectrum_spectrum_view()
        self.readings_view  = self.build_reading_reading_view()

        self.stacked_views.addWidget(self.default_view)
        self.stacked_views.addWidget(self.dual_view)
        self.stacked_views.addWidget(self.dual_waveform_view)
        self.stacked_views.addWidget(self.dual_spectrum_view)
        self.stacked_views.addWidget(self.readings_view)
        self.main_layout.addWidget(self.stacked_views)

        bottom_buttons = QHBoxLayout()
        for label in ["Traces", "Param", "Control", "Auto", "Cursor"]:
            if label == "Traces":
                self.traces_button = QPushButton(label)
                self.traces_menu = QMenu()
                self.traces_menu.addAction("Readings + Waveform", lambda: self.switch_trace_mode(0))
                self.traces_menu.addAction("Waveform + Spectrum", lambda: self.switch_trace_mode(1))
                self.traces_menu.addAction("Waveform + Waveform", lambda: self.switch_trace_mode(2))
                self.traces_menu.addAction("Spectrum + spectrum", lambda: self.switch_trace_mode(3))
                self.traces_menu.addAction("Readings + Readings", lambda: self.switch_trace_mode(4))
                self.traces_menu.addAction
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
    
    def build_waveform_waveform_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.waveform_top = Figure(figsize=(6,3))
        self.canvas_waveform_top = FigureCanvas(self.waveform_top)
        self.ax_waveform_top = self.waveform_top.add_subplot(111)
        self.canvas_waveform_top.mpl_connect("button_press_event",self.on_waveform_click)
        layout.addWidget(self.canvas_waveform_top)

        self.waveform_bottom = Figure(figsize=(6,3))
        self.canvas_waveform_bottom =  FigureCanvas(self.waveform_bottom)
        self.ax_waveform_bottom = self.waveform_bottom.add_subplot(111)
        self.canvas_waveform_bottom.mpl_connect("button_press_event",self.on_waveform_click)
        layout.addWidget(self.canvas_waveform_bottom)

        return widget
    
    def build_spectrum_spectrum_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.ch0_window =  Figure(figsize=(6,3))
        self.ch0_fft = FigureCanvas(self.ch0_window)
        self.ax_fft1 = self.ch0_window.add_subplot(111)
        self.ch0_fft.mpl_connect("button_press_event",self.on_fft_click)
        layout.addWidget(self.ch0_fft)

        self.ch1_window = Figure(figsize=(6,3))
        self.ch1_fft = FigureCanvas(self.ch1_window)
        self.ax_fft2 = self.ch1_window.add_subplot(111)
        self.ch1_fft.mpl_connect("button_press_event",self.on_fft_click)
        layout.addWidget(self.ch1_fft)

        return widget
    
    def build_reading_reading_view(self):
        widget =  QWidget()
        layout = QVBoxLayout(widget)

        readings_layout = QHBoxLayout()
        self.acc_input0 = self.create_reading_box("Acc:", "g (peak)")
        self.vel_input0 = self.create_reading_box("Vel:", "mm/s (RMS)")
        self.disp_input0 = self.create_reading_box("Disp:", "µm (P-P)")
        self.freq_input0 = self.create_reading_box("Freq:", "Hz")

        for box in [self.acc_input0, self.vel_input0, self.disp_input0, self.freq_input0]:
            readings_layout.addLayout(box["layout"])
        layout.addLayout(readings_layout)

        readings_layout1 = QHBoxLayout()
        self.acc_input1 = self.create_reading_box("Acc:","g (peak)")
        self.vel_input1 = self.create_reading_box("Vel:", "mm/s(RMS)")
        self.disp_input1 = self.create_reading_box("Disp:","µm (P-P)")
        self.freq_input1 = self.create_reading_box("Freq:","Hz")

        for box1 in [self.acc_input1,self.vel_input1,self.disp_input1,self.freq_input1]:
            readings_layout1.addLayout(box["layout"])
        layout.addLayout(readings_layout1)

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

    def update_plot(self):
        results = self.daq.get_latest_waveform()  # list of 2 dicts
        if not results or len(results) != 2:
            return

        result_ch0 = results[0]
        result_ch1 = results[1]

        # Set time axis from CH0
        self.t = result_ch0["time"]
        y_label = ""
        
        # Quantity selection
        if self.selected_quantity == "Velocity":
            y_ch0 = result_ch0["velocity"]
            y_ch1 = result_ch1["velocity"]
            self.fft_mags_top = result_ch0["fft_mags_vel"]
            self.fft_mags_bottom = result_ch1["fft_mags_vel"]
            self.freqs = result_ch0["freqs_vel"]
            y_label = "Velocity (mm/s)"
        elif self.selected_quantity == "Displacement":
            y_ch0 = result_ch0["displacement"]
            y_ch1 = result_ch1["displacement"]
            self.fft_mags_top = result_ch0["fft_mags_disp"]
            self.fft_mags_bottom = result_ch1["fft_mags_disp"]
            self.freqs = result_ch0["fft_freqs_disp"]
            y_label = "Displacement (μm)"
        else:
            y_ch0 = result_ch0["acceleration"]
            y_ch1 = result_ch1["acceleration"]
            self.fft_mags_top = result_ch0["fft_mags"]
            self.fft_mags_bottom = result_ch1["fft_mags"]
            self.freqs = result_ch0["fft_freqs"]
            y_label = "Acceleration (g)"

        # Update Readings (from CH0)
        
        view_index = self.stacked_views.currentIndex()
        if view_index == 0:

            self.acc_input["input"].setText(f"{result_ch0['acc_peak']:.2f}")
            self.vel_input["input"].setText(f"{result_ch0['rms_fft']:.2f}")
            self.disp_input["input"].setText(f"{result_ch0['disp_pp']:.2f}")
            self.freq_input["input"].setText(f"{result_ch0['dom_freq']:.2f}")

            # Default single waveform view (use CH0)
            self.ax_waveform.clear()
            self.ax_waveform.plot(self.t, y_ch0)
            self.ax_waveform.set_title("Waveform")
            self.ax_waveform.set_xlabel("Time (s)")
            self.ax_waveform.set_ylabel(y_label)
            margin = (max(y_ch0) - min(y_ch0)) * 0.1 or 0.2
            self.ax_waveform.set_ylim(min(y_ch0) - margin, max(y_ch0) + margin)
            self.ax_waveform.grid(True)
            self.canvas_waveform.draw()
        elif view_index == 1:
            # Dual view: CH0 top, CH1 bottom
            self.ax_top.clear()
            self.ax_top.plot(self.t, y_ch0)
            self.ax_top.set_title("Waveform CH0")
            self.ax_top.set_xlabel("Time (s)")
            self.ax_top.set_ylabel(y_label)
            margin0 = (max(y_ch0) - min(y_ch0)) * 0.1 or 0.2
            self.ax_top.set_ylim(min(y_ch0) - margin0, max(y_ch0) + margin0)
            self.ax_top.grid(True)
            self.canvas_top.draw()

            self.ax_bottom.clear()
            self.ax_bottom.plot(self.freqs, self.fft_mags_top)
            self.ax_bottom.set_title("Spectrum CH1")
            self.ax_bottom.set_xlabel("Frequency (Hz)")
            self.ax_bottom.set_ylabel("Magnitude (RMS)")
            self.ax_bottom.grid(True)
            self.canvas_bottom.draw()
        elif view_index == 2:
            self.ax_waveform_top.clear()
            self.ax_waveform_top.plot(self.t,y_ch0)
            self.ax_waveform_top.set_title("Waveform CH0")
            self.ax_waveform_top.set_xlabel("Time (s)")
            self.ax_waveform_top.set_ylabel(y_label)
            margin0 =(max(y_ch0)- min(y_ch0)) * 0.1 or 0.2
            self.ax_waveform_top.set_ylim(min(y_ch0) - margin0,max(y_ch0)+ margin0)
            self.ax_waveform_top.grid(True)
            self.canvas_waveform_top.draw()

            self.ax_waveform_bottom.clear()
            self.ax_waveform_bottom.plot(self.t,y_ch1)
            self.ax_waveform_bottom.set_title("Waveform CH1")
            self.ax_waveform_bottom.set_xlabel("Time (s)")
            self.ax_waveform_bottom.set_ylabel(y_label)
            margin1 = (max(y_ch1)- min(y_ch1)) * 0.1 or 0.2
            
            self.ax_waveform_bottom.grid(True)
            self.canvas_waveform_bottom.draw()

        elif view_index == 3:
            self.ax_fft1.clear()
            self.ax_fft1.plot(self.freqs,self.fft_mags_top)
            self.ax_fft1.set_title("Spectrum CH0")
            self.ax_fft1.set_xlabel("Frequency (Hz)")
            self.ax_fft1.set_ylabel("Magnitude")
            self.ax_fft1.grid(True)
            self.ch0_fft.draw()

            self.ax_fft2.clear()
            self.ax_fft2.plot(self.freqs,self.fft_mags_bottom)
            self.ax_fft2.set_title("Spectrum CH1")
            self.ax_fft2.set_xlabel("Frequency (Hz)")
            self.ax_fft2.set_ylabel("Magnitude")
            self.ax_fft2.grid(True)
            self.ch1_fft.draw()

        elif view_index == 4:
            self.acc_input0["input"].setText(f"{result_ch0['acc_peak']:.2f}")
            self.vel_input0["input"].setText(f"{result_ch0['rms_fft']:.2f}")
            self.disp_input0["input"].setText(f"{result_ch0['disp_pp']:.2f}")
            self.freq_input0["input"].setText(f"{result_ch0['dom_freq']:.2f}")

            self.acc_input1["input"].setText(f"{result_ch1['acc_peak']:.2f}")
            self.vel_input1["input"].setText(f"{result_ch1['rms_fft']:.2f}")
            self.disp_input1["input"].setText(f"{result_ch1['disp_pp']:.2f}")
            self.freq_input1["input"].setText(f"{result_ch1['dom_freq']:.2f}")





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

    