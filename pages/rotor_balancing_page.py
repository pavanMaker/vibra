
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
TACH_TIMEOUT = 1.0

FIXED_SAMPLE_RATE = 5000
FIXED_BUFFER_SIZE = 8192
SENSOR_CHANNEL = 1  # CH1 only

TRIGGER_DELAY_COMPENSATION = 180.0  # degrees, for 39-sample delay @ 5 kHz

STATE_WAIT_RPM     = 0
STATE_WAIT_STABLE  = 1
STATE_WAIT_TRIGGER = 2
STATE_SAMPLING     = 3
STATE_HOLD_LAST    = 4

# Rotation direction modes
DIR_WITH_ROTATION    = 0
DIR_AGAINST_ROTATION = 1


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
        arrow_len = max(0.3, min(self.amplitude, 1.0)) * r
        end_x = int(cx + arrow_len * math.cos(angle_rad))
        end_y = int(cy + arrow_len * math.sin(angle_rad))
        painter.drawLine(int(cx), int(cy), end_x, end_y)

        # Arrowhead
        head_size = 8
        for offset in [150, -150]:
            ha = angle_rad + math.radians(offset)
            painter.drawLine(end_x, end_y,
                             int(end_x + head_size * math.cos(ha)),
                             int(end_y + head_size * math.sin(ha)))

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

        stored = load_settings()
        self.sensitivity_v_per_g = stored.get("sensitivity_ch1", 0.1)

        self.state = STATE_WAIT_RPM
        self.last_rpm = None
        self.rpm_stable_count = 0
        self.locked_freq_1x = None
        self.last_tach_time = None
        self.last_valid_result = None
        self.analysis_buffer = []

        # Rotation direction (default: with rotation)
        self.rotation_direction = DIR_WITH_ROTATION

        self.daq = None
        self.tach = None
        self.read_timer = None
        self.hardware_initialized = False

        self.build_ui()

    # ===================== UI =====================
    def build_ui(self):
        self.setStyleSheet("background-color: black;")
        layout = QVBoxLayout(self)

        # -------- TOP BAR --------
        bar = QWidget()
        bar.setStyleSheet("background-color: #111111;")
        bar_l = QHBoxLayout(bar)
        back = QPushButton("⬅ Back")
        back.setStyleSheet("QPushButton{background-color:#333;color:white;padding:6px}"
                           "QPushButton:hover{background-color:#555}")
        back.clicked.connect(self.go_back)
        title = QLabel("Rotor Balancing")
        title.setStyleSheet("color:white;font-size:20px;font-weight:bold;")

        # -------- ROTATION DIRECTION TOGGLE --------
        self.dir_button = QPushButton("⟳ With Rotation")
        self.dir_button.setStyleSheet("""
            QPushButton {
                background-color: #006600;
                color: white;
                padding: 6px 14px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #008800;
            }
        """)
        self.dir_button.clicked.connect(self.toggle_rotation_direction)

        bar_l.addWidget(back)
        bar_l.addWidget(title)
        bar_l.addStretch()
        bar_l.addWidget(self.dir_button)
        layout.addWidget(bar)

        # -------- MAIN CONTENT --------
        top = QWidget()
        top_l = QHBoxLayout(top)

        self.canvas = RotorCanvas(self)
        self.canvas.setMinimumHeight(420)
        top_l.addWidget(self.canvas, 3)

        values = QWidget()
        values.setStyleSheet("QWidget{background-color:#000;border:1px solid #444}")
        grid = QGridLayout(values)

        title_style = "color:white;font-size:16px;"
        value_style = "color:#00FF00;font-size:18px;font-weight:bold;"

        labels = [("RPM:", "rpm_value", "--"),
                  ("1× Freq:", "freq_value", "-- Hz"),
                  ("Phase:", "phase_deg_label", "--°"),
                  ("Acc (1×):", "acc_label", "-- m/s²"),
                  ("Vel (1×):", "vel_label", "-- mm/s"),
                  ("Direction:", "dir_label", "With Rotation")]

        for row, (txt, attr, default) in enumerate(labels):
            lbl = QLabel(txt)
            lbl.setStyleSheet(title_style)
            val = QLabel(default)
            val.setStyleSheet(value_style)
            setattr(self, attr, val)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(val, row, 1)

        top_l.addWidget(values, 1)
        layout.addWidget(top)

        # -------- WAVEFORM PLOT --------
        self.fft_plot = pg.PlotWidget(title="Waveform View")
        self.fft_plot.setBackground('k')
        self.fft_plot.setLabel('bottom', 'Time (s)', color='white')
        self.fft_plot.setLabel('left', 'Acceleration (m/s²)', color='white')
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen(color='#00FF00', width=2))
        layout.addWidget(self.fft_plot)

    # ===================== ROTATION DIRECTION TOGGLE =====================
    def toggle_rotation_direction(self):
        if self.rotation_direction == DIR_WITH_ROTATION:
            self.rotation_direction = DIR_AGAINST_ROTATION
            self.dir_button.setText("⟲ Against Rotation")
            
            self.dir_button.setStyleSheet("""
                QPushButton {
                    background-color: #CC6600;
                    color: white;
                    padding: 6px 14px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #EE8800;
                }
            """)
            self.dir_label.setText("Against Rotation")
        else:
            self.rotation_direction = DIR_WITH_ROTATION
            self.dir_button.setText("⟳ With Rotation")
            self.dir_button.setStyleSheet("""
                QPushButton {
                    background-color: #006600;
                    color: white;
                    padding: 6px 14px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #008800;
                }
            """)
            self.dir_label.setText("With Rotation")

        # If we have a previous result, recalculate display with new direction
        if self.last_valid_result:
            self.update_ui(self.last_valid_result)

    # ===================== HW INIT =====================
    def showEvent(self, event):
        super().showEvent(event)
        if self.hardware_initialized:
            return

        self.daq = Mcc172Backend(
            sample_rate=FIXED_SAMPLE_RATE,
            buffer_size=FIXED_BUFFER_SIZE,
            channel=[SENSOR_CHANNEL],
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
        self.last_tach_time = time.time()
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

            if self.rpm_stable_count >= RPM_STABLE_COUNT_REQ:
                self.locked_freq_1x = rpm / 60.0
                self.freq_value.setText(f"{self.locked_freq_1x:.2f} Hz")
                self.arm_triggered_scan()
                self.state = STATE_WAIT_TRIGGER
            return

        if self.state == STATE_WAIT_TRIGGER:
            self.state = STATE_SAMPLING

    # ===================== ARM TRIGGER =====================
    def arm_triggered_scan(self):
        print(f"RPM stable @ {self.last_rpm} → arming scan  (1x={self.locked_freq_1x:.2f} Hz)")
        self.analysis_buffer.clear()

        try:
            self.daq.board.trigger_config(
                SourceType.LOCAL,
                TriggerModes.FALLING_EDGE,
            )
            self.daq.board.a_in_scan_start(
                channel_mask=(1 << SENSOR_CHANNEL),
                samples_per_channel=FIXED_BUFFER_SIZE,
                options=OptionFlags.EXTTRIGGER,
            )
            self.state = STATE_WAIT_TRIGGER
            self.canvas.is_sampling = True
            self.canvas.update()
        except Exception as e:
            print(f"Error arming scan: {e}")
            self.state = STATE_WAIT_RPM

    # ===================== READ =====================
    def read_daq_data(self):
        if self.state not in (STATE_WAIT_TRIGGER, STATE_SAMPLING):
            return
        if not self.is_tach_alive():
            print("Tach lost → hold")
            self.enter_hold_state()
            return

        try:
            ch0, ch1 = self.daq.read_data()
        except Exception as e:
            print(f"DAQ read error: {e}")
            return

        sensor_data = ch1 if len(ch1) > 0 else ch0

        if len(sensor_data) > 0:
            if self.state == STATE_WAIT_TRIGGER:
                print("Trigger fired → sampling")
                self.state = STATE_SAMPLING
            self.analysis_buffer.extend(sensor_data.tolist())
        if len(self.analysis_buffer) >= FIXED_BUFFER_SIZE:
            print(f"Buffer full ({len(self.analysis_buffer)} samples)")
            self.finish_scan()

    # ===================== FINISH =====================
    def finish_scan(self):
        print("Finishing scan …")
        try:
            self.daq.board.a_in_scan_stop()
            self.daq.board.a_in_scan_cleanup()
        except Exception as e:
            print(f"Cleanup warning: {e}")

        data = np.array(self.analysis_buffer[:FIXED_BUFFER_SIZE], dtype=np.float64)
        self.analysis_buffer.clear()

        if len(data) < 64:
            print("Not enough data")
            self.reset_cycle()
            return

        result = self.analyze_1x(data, self.daq.actual_rate, self.locked_freq_1x)
        self.last_valid_result = result
        self.update_ui(result)
        self.reset_cycle()

    # ===================== 1x ANALYSIS =====================
    def analyze_1x(self, volts, fs, freq_1x):
        N = len(volts)

        # Voltage → g → m/s²
        acc_g = volts / self.sensitivity_v_per_g
        acc_ms2 = acc_g * 9.80665
        acc_ms2 -= np.mean(acc_ms2)

        acc_display = acc_ms2.copy()

        # Hanning window
        window = np.hanning(N)
        acc_windowed = acc_ms2 * window
        coherent_gain = np.sum(window) / N

        # FFT
        fft_result = np.fft.rfft(acc_windowed)
        freqs = np.fft.rfftfreq(N, 1.0 / fs)

        # Single-sided peak-amplitude spectrum
        fft_mag = (2.0 / (N * coherent_gain)) * np.abs(fft_result)

        # Locate 1
        peak_idx = np.argmin(np.abs(freqs-freq_1x))

        C_1x = fft_result[peak_idx]
        actual_freq = freqs[peak_idx]
        acc_peak_1x = fft_mag[peak_idx]

        # ── PHASE CALCULATION ──
        # Raw phase from FFT
        raw_phase = np.degrees(np.arctan2(C_1x.imag, C_1x.real)) % 360.0

        # Add +60° compensation for 39-sample trigger delay
        phase_with = (raw_phase + TRIGGER_DELAY_COMPENSATION) % 360.0

        # Against rotation = 360 - with_rotation phase
        phase_against = (360.0 - phase_with) % 360.0

        # Velocity: v = a / ω
        vel_mm_s = (acc_peak_1x / (2 * np.pi * actual_freq) * 1000.0
                    if actual_freq > 0 else 0.0)

        print(f"── 1x result ──  freq={actual_freq:.2f} Hz  "
              f"acc={acc_peak_1x:.4f} m/s²  "
              f"phase_with={phase_with:.1f}°  phase_against={phase_against:.1f}°  "
              f"vel={vel_mm_s:.4f} mm/s")

        return {
            "phase_with": phase_with,
            "phase_against": phase_against,
            "amp": acc_peak_1x,
            "vel": vel_mm_s,
            "freq_1x_actual": actual_freq,
            "time_axis": np.linspace(0, N / fs, N, endpoint=False),
            "acc_signal": acc_display,
        }

    # ===================== UI UPDATE =====================
    def update_ui(self, r):
        # Select phase based on current rotation direction
        if self.rotation_direction == DIR_WITH_ROTATION:
            phase = r["phase_with"]
        else:
            phase = r["phase_against"]

        self.phase_deg_label.setText(f"{phase:.1f}°")
        self.acc_label.setText(f"{r['amp']:.3f} m/s²")
        self.vel_label.setText(f"{r['vel']:.3f} mm/s")

        self.canvas.phase_deg = phase
        self.canvas.amplitude = min(r["amp"] / 5.0, 1.0)
        self.canvas.update()

        self.fft_curve.setData(r["time_axis"], r["acc_signal"])

    # ===================== TACH HELPERS =====================
    def is_tach_alive(self):
        return self.last_tach_time and (time.time() - self.last_tach_time) < TACH_TIMEOUT

    def enter_hold_state(self):
        self.state = STATE_HOLD_LAST
        try:
            self.daq.board.a_in_scan_stop()
        except Exception:
            pass
        try:
            self.daq.board.a_in_scan_cleanup()
        except Exception:
            pass

    def reset_cycle(self):
        self.state = STATE_WAIT_RPM
        self.last_rpm = None
        self.rpm_stable_count = 0
        self.canvas.is_sampling = False
        self.canvas.update()

    # ===================== CLEANUP =====================
    def cleanup_hardware(self):
        # Stop timer FIRST to prevent race condition
        if self.read_timer:
            self.read_timer.stop()
            self.read_timer = None

        if self.tach:
            self.tach.cleanup()
            self.tach = None

        if self.daq:
            try:
                self.daq.board.a_in_scan_stop()
            except Exception:
                pass
            try:
                self.daq.board.a_in_scan_cleanup()
            except Exception:
                pass
            self.daq = None

        self.hardware_initialized = False

    def go_back(self):
        self.cleanup_hardware()
        self.main_window.stacked_widget.setCurrentIndex(0)