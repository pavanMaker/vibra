# rotor_balancing_page.py
import math
import numpy as np
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt6.QtGui import QPainter, QPen, QBrush, QFont
from PyQt6.QtCore import Qt, QTimer
from scipy.signal import detrend, butter, filtfilt, windows

from backend.mcc_backend import Mcc172Backend
from pages.tachometer import TachometerReader


# ==========================================================
# CANVAS
# ==========================================================
class RotorCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase_deg = 0.0
        self.amplitude = 0.0   # normalized 0..1
        self.is_sampling = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        try:
            size = int(min(self.width(), self.height() - 140))
            x = int((self.width() - size) / 2)
            y = 20

            # Circle
            painter.setPen(QPen(Qt.GlobalColor.white, 4))
            painter.drawEllipse(x, y, size, size)

            # Needle (FINAL phase only)
            painter.setPen(QPen(Qt.GlobalColor.red, 4))
            angle_rad = math.radians(self.phase_deg - 90)
            cx = x + size // 2
            cy = y + size // 2
            r = size // 2 - 10
            nx = int(cx + r * math.cos(angle_rad))
            ny = int(cy + r * math.sin(angle_rad))
            painter.drawLine(cx, cy, nx, ny)

            # Status text
            painter.setFont(QFont("Arial", 14))
            painter.setPen(QPen(Qt.GlobalColor.white))
            text = "Sampling..." if self.is_sampling else "Waiting for detection..."
            painter.drawText(x, y + size // 2 + 20, size, 30,
                             Qt.AlignmentFlag.AlignCenter, text)

            # Amplitude bar
            painter.setBrush(QBrush(Qt.GlobalColor.green))
            painter.setPen(Qt.PenStyle.NoPen)
            bar_w = size
            bar_h = 18
            bar_x = int((self.width() - bar_w) / 2)
            bar_y = y + size + 28
            fill_w = int(bar_w * max(0.0, min(1.0, self.amplitude)))
            painter.drawRect(bar_x, bar_y, fill_w, bar_h)

            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(bar_x, bar_y, bar_w, bar_h)

        finally:
            painter.end()


# ==========================================================
# ROTOR PAGE
# ==========================================================
class RotorPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        # DAQ
        self.daq = Mcc172Backend()
        self.daq.setup()

        # Tachometer
        self.tach = TachometerReader()
        self.current_rpm = 0.0

        self.tach.rpm_updated.connect(self.on_rpm_update)
        self.tach.first_pulse_detected.connect(self.on_sampling_started)

        self.waiting_for_rpm = False
        self.sampling_done = False

        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.create_header()

        top = QWidget()
        top_layout = QHBoxLayout(top)

        self.canvas = RotorCanvas(self)
        self.canvas.setMinimumHeight(420)
        top_layout.addWidget(self.canvas, stretch=3)

        values = QWidget()
        grid = QGridLayout(values)

        self.rpm_label = QLabel("RPM: --")
        self.freq_label = QLabel("1× Freq: -- Hz")
        self.phase_deg_label = QLabel("Phase (deg): --")
        self.phase_rad_label = QLabel("Phase (rad): --")
        self.amp_label = QLabel("Acc (pk): -- m/s²")

        for lbl in [
            self.rpm_label, self.freq_label,
            self.phase_deg_label, self.phase_rad_label,
            self.amp_label
        ]:
            lbl.setStyleSheet("font-size: 15px; color: black")

        grid.addWidget(self.rpm_label, 0, 0)
        grid.addWidget(self.freq_label, 1, 0)
        grid.addWidget(self.phase_deg_label, 2, 0)
        grid.addWidget(self.phase_rad_label, 3, 0)
        grid.addWidget(self.amp_label, 4, 0)

        top_layout.addWidget(values, stretch=1)
        self.layout.addWidget(top)

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(40)

    # ------------------------------------------------------
    def create_header(self):
        bar = QWidget()
        bar.setStyleSheet("background-color:#1e1e1e;")
        layout = QHBoxLayout(bar)

        back = QPushButton("⬅ Back")
        back.clicked.connect(self.go_back)
        back.setStyleSheet("background:#444;color:white;padding:6px")

        title = QLabel("Rotor Balancing")
        title.setStyleSheet("color:white;font-size:18px;font-weight:bold")

        layout.addWidget(back)
        layout.addWidget(title)
        layout.addStretch()
        self.layout.addWidget(bar)

    def go_back(self):
        self.main_window.stacked_widget.setCurrentIndex(0)

    # ------------------------------------------------------
    def on_rpm_update(self, rpm):
        print(f"🟢 RotorPage RPM received: {rpm:.2f}")

        self.current_rpm = rpm
        self.rpm_label.setText(f"RPM: {rpm:.1f}")
        self.freq_label.setText(f"1× Freq: {rpm/60.0:.2f} Hz")

        if self.waiting_for_rpm and not self.sampling_done:
            self.start_rotor_sampling()

    # ------------------------------------------------------
    def on_sampling_started(self):
        print("🟡 First pulse detected → arming sampling")

        self.waiting_for_rpm = True
        self.canvas.is_sampling = True
        self.canvas.update()
    # ------------------------------------------------------
    def start_rotor_sampling(self):
        print("🟢 RPM valid → starting acquisition")

        self.waiting_for_rpm = False
        self.sampling_done = True

        try:
            self.daq.start_acquisition()
            ch0,ch1 = self.daq.read_data()

            result = RotorPage.phase_calculation(
                raw_voltage=ch1,
                fs=self.daq.actual_rate,
                sensitivity_v_per_g=0.1,
                rpm=self.current_rpm
            )

            self.phase_deg_label.setText(f"Phase (deg): {result['phase_deg']:.2f}")
            self.phase_rad_label.setText(f"Phase (rad): {result['phase_rad']:.4f}")
            self.amp_label.setText(f"Acc (pk): {result['acc_amp_pk']:.3f} m/s²")

            self.canvas.phase_deg = result["phase_deg"]
            self.canvas.amplitude = min(result["acc_amp_pk"] / 10.0, 1.0)

        finally:
            self.daq.stop_scan()
            self.canvas.is_sampling = False
            self.canvas.update()

     
    @staticmethod
    def phase_calculation(raw_voltage,fs,sensitivity_v_per_g,rpm,fmin=3,fmax_hz=1150):
       
        if raw_voltage is None or len(raw_voltage) < 256:
            return {"phase_deg": 0.0, "phase_rad": 0.0, "acc_amp_pk": 0.0}

        if rpm <= 0:
            return {"phase_deg": 0.0, "phase_rad": 0.0, "acc_amp_pk": 0.0}

        # ------------------------------
        # Convert voltage → acceleration
        # ------------------------------
        acc_g = raw_voltage / sensitivity_v_per_g
        acc_g = detrend(acc_g)                  # remove DC & drift
        acc_ms2 = acc_g * 9.80665               # g → m/s²

        # ------------------------------
        # Band-pass filter (broadband)
        # ------------------------------
        nyq = fs / 2.0
        fmax = min(fmax_hz, nyq * 0.95)

        b, a = butter(
            N=4,
            Wn=[fmin / nyq, fmax / nyq],
            btype="band"
        )
        acc_filt = filtfilt(b, a, acc_ms2)

        # ------------------------------
        # 1× frequency
        # ------------------------------
        freq_1x = rpm / 60.0

        # ------------------------------
        # Windowing (CRITICAL for phase)
        # ------------------------------
        win = windows.hann(len(acc_filt))
        acc_win = acc_filt * win

        # ------------------------------
        # FFT
        # ------------------------------
        fft_acc = np.fft.rfft(acc_win)
        freqs = np.fft.rfftfreq(len(acc_win), d=1.0 / fs)

        # Find closest bin to 1×
        idx = np.argmin(np.abs(freqs - freq_1x))

        # ------------------------------
        # Phase (CORRECT)
        # ------------------------------
        phase_rad = np.angle(fft_acc[idx])
        phase_deg = (np.degrees(phase_rad) + 360.0) % 360.0

        # ------------------------------
        # Amplitude (CORRECT 1× ONLY)
        # ------------------------------
        fft_mag = np.abs(fft_acc[idx])

        coherent_gain = 0.5        # Hann window
        N = len(acc_win)

        acc_amp_pk = (2.0 * fft_mag) / (N * coherent_gain)

        return {
            "phase_deg": phase_deg,
            "phase_rad": phase_rad,
            "acc_amp_pk": acc_amp_pk
        }


    def update_animation(self):
        self.canvas.update()
