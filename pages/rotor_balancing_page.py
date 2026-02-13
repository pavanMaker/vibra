
# rotor_balancing_page.py
import math
import time
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout
)
from PyQt6.QtGui import QPainter, QPen, QBrush, QFont
from PyQt6.QtCore import Qt, QTimer

from daqhats import OptionFlags, TriggerModes, SourceType
from backend.mcc_backend import Mcc172Backend
from pages.tachometer import TachometerReader
from pages.settings_manager import load_settings


# ===================== CONSTANTS =====================
RPM_DIFF_LIMIT = 3
RPM_STABLE_COUNT_REQ = 2
TACH_TIMEOUT = 0.5

STATE_WAIT_RPM     = 0
STATE_WAIT_STABLE  = 1
STATE_WAIT_TRIGGER = 2
STATE_SAMPLING     = 3
STATE_HOLD_LAST    = 4


# ===================== ROTOR CANVAS =====================
class RotorCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase_deg = 0.0
        self.amplitude = 0.0
        self.is_sampling = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = int(min(self.width(), self.height() - 140))
        x = (self.width() - size) // 2
        y = 20
        cx, cy = x + size // 2, y + size // 2
        r = size // 2 - 10

        painter.setPen(QPen(Qt.GlobalColor.gray, 1, Qt.PenStyle.DashLine))
        painter.drawLine(cx - r, cy, cx + r, cy)
        painter.drawLine(cx, cy - r, cx, cy + r)

        painter.setPen(QPen(Qt.GlobalColor.white, 4))
        painter.drawEllipse(x, y, size, size)

        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.setPen(QPen(Qt.GlobalColor.yellow))
        painter.drawText(cx - 15, cy - r - 20, "0°")
        painter.drawText(cx + r + 10, cy + 5, "90°")
        painter.drawText(cx - 15, cy + r + 25, "180°")
        painter.drawText(cx - r - 30, cy + 5, "270°")

        painter.setPen(QPen(Qt.GlobalColor.red, 4))
        angle_rad = math.radians(self.phase_deg - 90)

        end_x = int(cx + r * math.cos(angle_rad))
        end_y = int(cy + r * math.sin(angle_rad))

        painter.drawLine(int(cx), int(cy), end_x, end_y)


        painter.setBrush(QBrush(Qt.GlobalColor.red))
        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

        painter.setFont(QFont("Arial", 14))
        painter.setPen(Qt.GlobalColor.white)
        status = "Sampling..." if self.is_sampling else "Waiting for Tach..."
        painter.drawText(x, y + size + 10, size, 25,
                         Qt.AlignmentFlag.AlignCenter, status)

        painter.end()


# ===================== ROTOR PAGE =====================
class RotorPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        # ---------- SETTINGS ----------
        stored = load_settings()
        self.sensitivity_v_per_g = stored.get("sensitivity_ch1", 0.1)
        self.max_samples = 8192
        self.sample_rate = 5000

        # ---------- STATE ----------
        self.state = STATE_WAIT_RPM
        self.last_rpm = None
        self.rpm_stable_count = 0
        self.locked_freq_1x = None
        self.last_tach_time = None
        self.last_valid_result = None

        self.analysis_buffer = []

        # ---------- LAZY HARDWARE ----------
        self.daq = None
        self.tach = None
        self.read_timer = None
        self.hardware_initialized = False

        # ---------- UI ----------
        self.build_ui()

    # ===================== UI =====================
    def build_ui(self):
    # -------- MAIN LAYOUT --------
        self.setStyleSheet("background-color: black;")
        layout = QVBoxLayout(self)

        # -------- TOP BAR --------
        bar = QWidget()
        bar.setStyleSheet("background-color: #111111;")
        bar_l = QHBoxLayout(bar)

        back = QPushButton("⬅ Back")
        back.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        back.clicked.connect(self.go_back)

        title = QLabel("Rotor Balancing")
        title.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: bold;
        """)

        bar_l.addWidget(back)
        bar_l.addWidget(title)
        bar_l.addStretch()

        layout.addWidget(bar)

        # -------- MAIN CONTENT --------
        top = QWidget()
        top_l = QHBoxLayout(top)

        # -------- ROTOR CANVAS --------
        self.canvas = RotorCanvas(self)
        self.canvas.setMinimumHeight(420)
        top_l.addWidget(self.canvas, 3)

        # -------- VALUE PANEL --------
        values = QWidget()
        values.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border: 1px solid #444444;
            }
        """)
        grid = QGridLayout(values)

        # Labels (White)
        self.rpm_label = QLabel("RPM:")
        self.freq_label = QLabel("1× Freq:")
        self.phase_label = QLabel("Phase:")
        self.acc_title_label = QLabel("Acc:")
        self.vel_title_label = QLabel("Vel:")

        # Values (Green Digital Style)
        self.rpm_value = QLabel("--")
        self.freq_value = QLabel("-- Hz")
        self.phase_deg_label = QLabel("--°")
        self.acc_label = QLabel("--")
        self.vel_label = QLabel("--")

        title_style = "color: white; font-size: 16px;"
        value_style = """
            color: #00FF00;
            font-size: 18px;
            font-weight: bold;
        """

        for lbl in [
            self.rpm_label,
            self.freq_label,
            self.phase_label,
            self.acc_title_label,
            self.vel_title_label
        ]:
            lbl.setStyleSheet(title_style)

        for val in [
            self.rpm_value,
            self.freq_value,
            self.phase_deg_label,
            self.acc_label,
            self.vel_label
        ]:
            val.setStyleSheet(value_style)

        # Add to grid
        grid.addWidget(self.rpm_label, 0, 0)
        grid.addWidget(self.rpm_value, 0, 1)

        grid.addWidget(self.freq_label, 1, 0)
        grid.addWidget(self.freq_value, 1, 1)

        grid.addWidget(self.phase_label, 2, 0)
        grid.addWidget(self.phase_deg_label, 2, 1)

        grid.addWidget(self.acc_title_label, 3, 0)
        grid.addWidget(self.acc_label, 3, 1)

        grid.addWidget(self.vel_title_label, 4, 0)
        grid.addWidget(self.vel_label, 4, 1)

        top_l.addWidget(values, 1)
        layout.addWidget(top)

        # -------- WAVEFORM / SIGNAL PLOT --------
        self.fft_plot = pg.PlotWidget(title="Waveform View")
        self.fft_plot.setBackground('k')  # Black background
        self.fft_plot.setLabel('bottom', 'Time (s)', color='white')
        self.fft_plot.setLabel('left', 'Acceleration (m/s²)', color='white')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)

        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen(color='#00FF00', width=2))

        layout.addWidget(self.fft_plot)


    # ===================== LAZY LOAD =====================
    def showEvent(self, event):
        super().showEvent(event)
        if self.hardware_initialized:
            return

        self.daq = Mcc172Backend(
            
            sample_rate=5000,
            buffer_size=8192,
        )
        self.daq.setup()

        self.tach = TachometerReader()
        self.tach.rpm_updated.connect(self.on_rpm_update)

        self.read_timer = QTimer(self)
        self.read_timer.timeout.connect(self.read_daq_data)
        self.read_timer.start(50)

        self.hardware_initialized = True

    def hideEvent(self, event):
        super().hideEvent(event)
        self.cleanup_hardware()

    # ===================== RPM =====================
    def on_rpm_update(self, rpm):
        self.last_tach_time = time.time()#doubt
        rpm = int(round(rpm))
        self.rpm_value.setText(f"{rpm}")


        if self.state == STATE_HOLD_LAST:
            self.reset_cycle()
            return

        if self.state == STATE_WAIT_RPM:
            self.last_rpm = rpm
            self.state = STATE_WAIT_STABLE
            return

        if self.state == STATE_WAIT_STABLE:
            if abs(rpm - self.last_rpm) < RPM_DIFF_LIMIT:
                self.rpm_stable_count += 1
            else:
                self.rpm_stable_count = 0

            self.last_rpm = rpm

            if self.rpm_stable_count >= 2:
                self.locked_freq_1x = rpm / 60.0
                self.freq_value.setText(f"{self.locked_freq_1x:.2f} Hz")
                self.arm_triggered_scan()
                self.state = STATE_WAIT_TRIGGER
            return

        if self.state == STATE_WAIT_TRIGGER:
            self.state = STATE_SAMPLING

    # ===================== ARM TRIGGER =====================
    def arm_triggered_scan(self):
        print("RPM stabilized → ARM TRIGGERED SCAN")

        self.analysis_buffer.clear()# doubt

        try:
            # Configure trigger on falling edge
            self.daq.board.trigger_config(
                SourceType.LOCAL,
                TriggerModes.FALLING_EDGE
            )
            print("Trigger configured: LOCAL, FALLING_EDGE")

            # Start finite scan on CH1
            self.daq.board.a_in_scan_start(
                channel_mask=0b11,                
                samples_per_channel=8192,         
                options=OptionFlags.EXTTRIGGER    
            )

            print("Scan armed and waiting for trigger")

            self.state = STATE_WAIT_TRIGGER
            self.canvas.is_sampling = True
            self.canvas.update()

        except Exception as e:
            print("Error arming scan:", e)
            self.state = STATE_WAIT_RPM


    # ===================== READ DATA =====================
    # def read_daq_data(self):
    #     if self.state not in (STATE_WAIT_TRIGGER, STATE_SAMPLING):
    #         return

    #     if not self.is_tach_alive():
    #         self.enter_hold_state()
    #         return

    #     ch0, ch1 = self.daq.read_data()

    #     if len(ch1) > 0:
    #         if self.state == STATE_WAIT_TRIGGER:
    #             # 🔑 FIRST DATA = TRIGGER FIRED
    #             self.state = STATE_SAMPLING

    #         self.analysis_buffer.extend(ch1)

    #     if len(self.analysis_buffer) >= self.max_samples:
    #         self.finish_scan()

    def read_daq_data(self):
        if self.state not in (STATE_WAIT_TRIGGER, STATE_SAMPLING):
            return

        if not self.is_tach_alive():
            self.enter_hold_state()
            return

        ch0, ch1 = self.daq.read_data()

        # 🔴 We only care about CH1
        if len(ch1) > 0:
            if self.state == STATE_WAIT_TRIGGER:
                print("Trigger fired → Sampling started")
                self.state = STATE_SAMPLING

            self.analysis_buffer.extend(ch1)

        if len(self.analysis_buffer) >= 8000:
            print("buffer is filled")
            self.finish_scan()



 

    # ===================== FINISH =====================
    def finish_scan(self):
        print("finish called")
        self.daq.board.a_in_scan_stop()
        self.daq.board.a_in_scan_cleanup()

        data = np.array(self.analysis_buffer[:self.max_samples])
        self.analysis_buffer.clear()

        result = self.analyze_1x(data, self.daq.actual_rate, self.locked_freq_1x)
        self.last_valid_result = result
        self.update_ui(result)
        self.reset_cycle()

    # ===================== FFT =====================
    def analyze_1x(self, volts, fs, freq_1x):
        acc = (volts / self.sensitivity_v_per_g) * 9.80665
        acc_det = acc - np.mean(acc)

       

        window = np.hanning(len(acc))
        acc_det *= window
        window_gain = np.sum(window) / len(window)

        fft = np.fft.rfft(acc_det)
        freqs = np.fft.rfftfreq(len(acc_det), 1 / fs)

        fft_mag = (2 / (len(acc_det) * window_gain)) * np.abs(fft)
        print("FFT computed:", fft_mag)

        idx = np.argmin(np.abs(freqs - freq_1x))
        C = fft[idx]

        phase = (np.degrees(np.arctan2(C.imag, C.real)) + 360) % 360
        acc_pk = fft_mag[idx]
        print(f"1x Freq: {freq_1x:.2f} Hz, Phase: {phase:.1f}°, Acc: {acc_pk:.2f} m/s²")
        vel_pk = (acc_pk / (2 * np.pi * freq_1x)) * 1000 if freq_1x > 0 else 0.0


        return {"phase": phase, "amp": acc_pk, "vel": vel_pk,"time_axis": np.linspace(0, len(acc) / fs, len(acc), endpoint=False), "acc_signal": acc_det}

    # ===================== UI UPDATE =====================
    def update_ui(self, r):
        self.phase_deg_label.setText(f"Phase: {r['phase']:.1f}°")
        self.acc_label.setText(f"Acc: {r['amp']:.2f}")
        self.canvas.phase_deg = r["phase"]
        self.canvas.amplitude = min(r["amp"] / 10.0, 1.0)
        self.canvas.update()

        self.fft_curve.setData(
        r["time_axis"],
        r["acc_signal"]
    )

    # ===================== TACH LOSS =====================
    def is_tach_alive(self):
        return self.last_tach_time and (time.time() - self.last_tach_time) < TACH_TIMEOUT

    def enter_hold_state(self):
        self.state = STATE_HOLD_LAST
        try:
            self.daq.board.a_in_scan_stop()
            self.daq.board.a_in_scan_cleanup()
        except:
            pass

        if self.last_valid_result:
            self.update_ui(self.last_valid_result)

        self.canvas.is_sampling = False
        self.canvas.update()

    # ===================== RESET =====================
    def reset_cycle(self):
        
        self.state = STATE_WAIT_RPM
        self.last_rpm = None
        self.rpm_stable_count = 0
        self.canvas.is_sampling = False
        self.canvas.update()

    # ===================== CLEANUP =====================
    def cleanup_hardware(self):
        if self.read_timer:
            self.read_timer.stop()
            self.read_timer = None

        if self.tach:
            self.tach.cleanup()
            self.tach = None

        if self.daq:
            try:
                self.daq.board.a_in_scan_stop()
                self.daq.board.a_in_scan_cleanup()
            except:
                pass
            self.daq = None

        self.hardware_initialized = False

    def go_back(self):
        self.cleanup_hardware()
        self.main_window.stacked_widget.setCurrentIndex(0)
