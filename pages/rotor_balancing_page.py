# rotor_balancing_page.py
import math
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout
)
from PyQt6.QtGui import QPainter, QPen, QBrush, QFont
from PyQt6.QtCore import Qt, QTimer

from backend.mcc_backend import Mcc172Backend
from pages.tachometer import TachometerReader
from pages.settings_manager import load_settings
import time



# ======================================================
# CANVAS WITH DEGREE MARKINGS
# ======================================================
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

        # Draw crosshair
        painter.setPen(QPen(Qt.GlobalColor.gray, 1, Qt.PenStyle.DashLine))
        painter.drawLine(cx - r, cy, cx + r, cy)
        painter.drawLine(cx, cy - r, cx, cy + r)

        # Draw main circle
        painter.setPen(QPen(Qt.GlobalColor.white, 4))
        painter.drawEllipse(x, y, size, size)

        # Draw degree markings
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.setPen(QPen(Qt.GlobalColor.yellow))
        
        degree_positions = [
            (0, cx, cy - r - 20, "0°"),
            (90, cx + r + 15, cy, "90°"),
            (180, cx, cy + r + 20, "180°"),
            (270, cx - r - 25, cy, "270°")
        ]
        
        for angle, px, py, label in degree_positions:
            painter.drawText(int(px - 15), int(py + 5), label)

        # Draw phase vector (red line)
        painter.setPen(QPen(Qt.GlobalColor.red, 4))
        angle_rad = math.radians(self.phase_deg - 90)
        end_x = cx + r * math.cos(angle_rad)
        end_y = cy + r * math.sin(angle_rad)
        painter.drawLine(int(cx), int(cy), int(end_x), int(end_y))

        # Draw center dot
        painter.setBrush(QBrush(Qt.GlobalColor.red))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - 5, cy - 5, 10, 10)

        # Status text
        painter.setFont(QFont("Arial", 14))
        painter.setPen(QPen(Qt.GlobalColor.white))
        status = "Sampling..." if self.is_sampling else "Waiting for Tach..."
        painter.drawText(x, y + size + 10, size, 25,
                         Qt.AlignmentFlag.AlignCenter, status)

        # Amplitude bar
        bar_y = y + size + 45
        bar_w = size
        bar_h = 16
        fill = int(bar_w * max(0.0, min(1.0, self.amplitude)))

        painter.setBrush(QBrush(Qt.GlobalColor.green))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(x, bar_y, fill, bar_h)

        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(x, bar_y, bar_w, bar_h)

        painter.end()


# ======================================================
# ROTOR PAGE WITH DEBUG
# ======================================================
class RotorPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        # ================= STATE =================
        self.collecting = False
        self.current_rpm = 0.0
        self.current_freq_1x = 0.0
        self.rotation_count = 0
        self.rpm_ready = False
        self.last_tach_time = None
        self.last_buffer_start_time = None
        # self.buffer_ready = False

        self.last_valid_result = None
        self.samples_collected = 0

        # Load sensitivity from settings
        stored = load_settings()
        self.sensitivity_v_per_g = stored.get("sensitivity_ch1", 0.1)
        self.no_of_samples = stored.get("buffer_size", 8192)
        print(f"🔧 Loaded sensitivity: {self.sensitivity_v_per_g} V/g")

        self.analysis_buffer = []
        self.max_samples = self.no_of_samples

        # AVERAGING LISTS
        self.phase_history = []
        self.acc_history = []
        self.vel_history = []

        # ================= WATCHDOG =================
        self.tach_timeout_ms = 500
        self.watchdog_timer = QTimer()
        self.watchdog_timer.setSingleShot(True)
        self.watchdog_timer.timeout.connect(self.on_tach_timeout)

        # ✅ LAZY INITIALIZATION
        self.daq = None
        self.tach = None
        self.read_timer = None
        self.hardware_initialized = False

        # ================= UI =================
        self.build_ui()
        
        print("✅ RotorPage __init__ complete")

    # ======================================================
    # LAZY INITIALIZATION
    # ======================================================
    def showEvent(self, event):
        """Called when page becomes visible"""
        super().showEvent(event)
        
        if not self.hardware_initialized:
            print("\n" + "="*60)
            print(" ROTOR BALANCING PAGE OPENED - INITIALIZING HARDWARE")
            print("="*60)
            self.initialize_hardware()
    
    def initialize_hardware(self):
        """Initialize DAQ and Tachometer"""
        try:
            # Load settings
            stored = load_settings()
            No_of_sampels = stored.get("buffer_size", 8192)
            saved_sample_rate = stored.get("sample_rate", None)
            if saved_sample_rate is None:
                selected_fmax_hz = stored.get("selected_fmax_hz", 500.0)
                saved_sample_rate = int(selected_fmax_hz * 2.56)
            
            print(f" Sample rate: {saved_sample_rate} Hz")

            # ================= DAQ =================
            print("Initializing DAQ...")
            self.daq = Mcc172Backend(
                buffer_size=No_of_sampels,
                sample_rate=saved_sample_rate
            )
            self.daq.setup()
            print(f"DAQ initialized (actual rate: {self.daq.actual_rate} Hz)")

            # ================= TACH =================
            print("Initializing Tachometer...")
            self.tach = TachometerReader()
            # self.tach.first_pulse_detected.connect(self.on_first_pulse)
            self.tach.rpm_updated.connect(self.on_rpm_update)
            # self.tach.rotation_complete.connect(self.on_rotation_complete)
            print(" Tachometer initialized")

            # ================= DAQ READ TIMER =================
            print("Starting DAQ read timer...")
            self.read_timer = QTimer()
            self.read_timer.timeout.connect(self.read_daq_data)
            self.read_timer.start(50)
            print("DAQ read timer started (50ms interval)")

            self.hardware_initialized = True
            print("="*60)
            print("ROTOR BALANCING HARDWARE READY!")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"ERROR initializing hardware: {e}")
            import traceback
            traceback.print_exc()
            self.hardware_initialized = False
    
    def hideEvent(self, event):
        """Called when page is hidden"""
        super().hideEvent(event)
        
        print("\n Rotor page hidden - stopping collection")
        if self.collecting:
            self.collecting = False
            self.analysis_buffer.clear()
        
        if self.daq:
            try:
                self.daq.stop_scan()
                print("DAQ stopped")
            except Exception as e:
                print(f"Error stopping DAQ: {e}")
    # ======================================================
    # UI
    # ======================================================
    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        bar = QWidget()
        bar.setStyleSheet("background:#1e1e1e")
        h = QHBoxLayout(bar)

        back = QPushButton("⬅ Back")
        back.clicked.connect(self.go_back)
        back.setStyleSheet("background:#444;color:white;padding:6px")

        title = QLabel("Rotor Balancing")
        title.setStyleSheet("color:white;font-size:18px;font-weight:bold")

        h.addWidget(back)
        h.addWidget(title)
        h.addStretch()
        layout.addWidget(bar)

        top = QWidget()
        h = QHBoxLayout(top)

        self.canvas = RotorCanvas(self)
        self.canvas.setMinimumHeight(420)
        h.addWidget(self.canvas, 3)

        values = QWidget()
        g = QGridLayout(values)

        self.rpm_label = QLabel("RPM: --")
        self.freq_label = QLabel("1× Freq: -- Hz")
        self.phase_deg_label = QLabel("Phase: --°")
        self.acc_label = QLabel("Acc: -- m/s²")
        self.vel_label = QLabel("Vel: -- mm/s")
        self.rotation_label = QLabel("Rotations: 0")
        self.status_label = QLabel("Status: Waiting...")

        labels = [
            self.rpm_label, self.freq_label,
            self.phase_deg_label, self.acc_label,
            self.vel_label, self.rotation_label,
            self.status_label
        ]

        for i, lbl in enumerate(labels):
            lbl.setStyleSheet("font-size:15px")
            g.addWidget(lbl, i, 0)

        h.addWidget(values, 1)
        layout.addWidget(top)

    # ======================================================
    # NAVIGATION
    # ======================================================
    def go_back(self):
        print("\n🔙 Going back to main page")
        self.collecting = False
        self.analysis_buffer.clear()
        
        if self.tach:                                                                             
            self.tach.cleanup()
        
        if self.daq:
            try:
                self.daq.stop_scan()
            except:
                pass
        
        self.main_window.stacked_widget.setCurrentIndex(0)

    # ======================================================
    # TACH EVENTS
    # ======================================================
    def on_first_pulse(self):
        pass

    def on_rpm_update(self, rpm):
        self.last_tach_time = time.perf_counter()
        self.current_rpm = rpm
        self.current_freq_1x = rpm / 60.0

        self.rpm_label.setText(f"RPM: {rpm:.1f}")
        self.freq_label.setText(f"1× Freq: {self.current_freq_1x:.2f} Hz")

        # ---------- RPM GATE ----------
        if rpm <= 10:
            self.collecting = False
            # self.buffer_ready = False
            self.analysis_buffer.clear()
            self.last_valid_result = None

            if self.daq:
                self.daq.stop_scan()

            self.canvas.is_sampling = False
            self.canvas.update()
            self.status_label.setText("Status: Waiting for RPM > 10")
            return

        # ---------- START SAMPLING ----------
        if not self.collecting:
            print(" RPM > 10 → START DAQ")

            self.collecting = True
            # self.buffer_ready = False
            self.samples_collected = 0
            self.analysis_buffer.clear()

            self.daq.start_acquisition()

            self.canvas.is_sampling = True
            self.canvas.update()
            # 

        self.watchdog_timer.start(self.tach_timeout_ms)

  

    def on_rotation_complete(self):
        pass

    def process_full_buffer(self):
        pass

    def on_tach_timeout(self):
       

        # Stop sampling
        self.collecting = False
        self.analysis_buffer.clear()

        if self.daq:
            self.daq.stop_scan()

        # Stop sampling indicator ONLY
        self.canvas.is_sampling = False
        self.canvas.update()

       
    # ======================================================
    # DAQ READ
    # ======================================================

    def read_daq_data(self):
        if not self.collecting or not self.daq:
            return

        _, ch1 = self.daq.read_data()
        if len(ch1) == 0:
            return
        
        if self.samples_collected == 0:
            self.last_buffer_start_time = time.perf_counter()

        # Fill buffer
        self.analysis_buffer.extend(ch1)
        self.samples_collected += len(ch1)

        # Full buffer reached
        if self.samples_collected >= self.max_samples:
            data = np.array(self.analysis_buffer[:self.max_samples])
            self.analysis_buffer.clear()
            self.samples_collected = 0

            result = self.analyze_1x(
                data,
                self.daq.actual_rate,
                self.current_freq_1x,
                self.sensitivity_v_per_g,
                self.last_buffer_start_time,
                self.last_tach_time
            
            )

            # HOLD until next buffer
            self.last_valid_result = result
            self.update_ui(result)



 

    # ======================================================
    # 1× EXTRACTION WITH DEBUG
    # ======================================================
    @staticmethod
    def analyze_1x(signal_volts, fs, freq_1x, sensitivity_v_per_g,
                buffer_start_time, tach_time):

        acc = (signal_volts / sensitivity_v_per_g) * 9.80665
        acc -= np.mean(acc)

        N = len(acc)
        fft = np.fft.rfft(acc)
        freqs = np.fft.rfftfreq(N, 1/fs)

        idx = int(round(freq_1x / (fs / N)))
        idx = max(1, min(idx, len(fft)-1))

        C = fft[idx]
        X = C.real
        Y = C.imag

        acc_pk = 2 * np.sqrt(X*X + Y*Y) / N
        phase_fft = np.arctan2(Y, X)   # radians

        # 🔥 PHASE CORRECTION USING TACH
        if buffer_start_time and tach_time:
            dt = buffer_start_time - tach_time
            phase_correction = 2 * np.pi * freq_1x * dt
            phase_corr = phase_fft - phase_correction
        else:
            phase_corr = phase_fft

        phase_deg = (np.degrees(phase_corr) + 360) % 360

        vel_pk = (acc_pk / (2*np.pi*freq_1x))*1000 if freq_1x > 0 else 0.0

        return {
            "phase_deg": phase_deg,
            "acc_pk": acc_pk,
            "vel_pk": vel_pk,
            "idx": idx,
            "bin_freq": freqs[idx],
            "bin_width": freqs[1] - freqs[0]
        }




    # ======================================================
    # UI UPDATE
    # ======================================================
    def update_ui(self, result):
        idx = result["idx"]
        bin_freq = result["bin_freq"]
        bin_width = result["bin_width"]

        self.phase_deg_label.setText(f"Phase: {result['phase_deg']:.1f}°")
        self.acc_label.setText(f"Acc: {result['acc_pk']:.2f} m/s²")
        self.vel_label.setText(f"Vel: {result['vel_pk']:.2f} mm/s")

        # 🔍 DEBUG DISPLAY
        self.rotation_label.setText(
            f"1× idx: {idx} | bin f: {bin_freq:.3f} Hz | Δf: {bin_width:.3f}"
        )

        self.canvas.phase_deg = result["phase_deg"]
        self.canvas.amplitude = min(result["acc_pk"] / 10.0, 1.0)
        self.canvas.update()


    # ======================================================
    # CLEANUP
    # ======================================================
    def closeEvent(self, event):
        print("\n RotorPage closing")
        self.collecting = False
        self.analysis_buffer.clear()
        
        if self.tach:
            self.tach.cleanup()
        
        if self.daq:
            try:
                self.daq.stop_scan()
            except:
                pass
        
        super().closeEvent(event)
