import pigpio
from PyQt6.QtCore import QObject, pyqtSignal


class TachometerReader(QObject):
    rpm_updated = pyqtSignal(float)
    first_pulse_detected = pyqtSignal()
    rotation_complete = pyqtSignal()   

    def __init__(self, gpio_pin=17, pulses_per_rev=1, debounce_us=5000):
        super().__init__()

        self.TACH_PIN = gpio_pin
        self.PULSES_PER_REV = pulses_per_rev
        self.DEBOUNCE_US = debounce_us

        self.first_pulse = False
        self.last_tick = None

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError(" pigpio daemon not running (run: sudo pigpiod)")

        print(f"✅ Tachometer initialized on GPIO {self.TACH_PIN}")

        self.pi.set_mode(self.TACH_PIN, pigpio.INPUT)
        self.pi.set_pull_up_down(self.TACH_PIN, pigpio.PUD_UP)

        self.cb = self.pi.callback(
            self.TACH_PIN,
            pigpio.FALLING_EDGE,
            self._pulse_callback
        )

    # ======================================================
    # TACH CALLBACK (HARD REAL-TIME)
    # ======================================================
    def _pulse_callback(self, gpio, level, tick):
        # pigpio safety check
        if level != 0:
            return

        # ------------------------------
        # FIRST PULSE (ARM SYSTEM)
        # ------------------------------
        if not self.first_pulse:
            self.first_pulse = True
            self.last_tick = tick
            self.first_pulse_detected.emit()
            return

        # ------------------------------
        # TIME BETWEEN PULSES
        # ------------------------------
        delta_us = pigpio.tickDiff(self.last_tick, tick)

        # Debounce / noise rejection
        if delta_us < self.DEBOUNCE_US:
            print("⚠️ Pulse ignored (debounce)")
            return

        # ------------------------------
        # RPM CALCULATION
        # ------------------------------
        rpm = (60_000_000.0 / delta_us) / self.PULSES_PER_REV
       # print(f"🟡 RPM = {rpm:.2f}")

        self.rpm_updated.emit(rpm)

        # ------------------------------
        # ROTATION COMPLETE (🔥 KEY SIGNAL)
        # ------------------------------
        self.rotation_complete.emit()

        # Update timestamp
        self.last_tick = tick

    # ======================================================
    def cleanup(self):
        if self.cb:
            self.cb.cancel()
        if self.pi:
            self.pi.stop()
        print("🧹 Tachometer cleaned up")
