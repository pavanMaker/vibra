import pigpio
from PyQt6.QtCore import QObject, pyqtSignal

class TachometerReader(QObject):
    rpm_updated = pyqtSignal(float)
    first_pulse_detected = pyqtSignal()

    def __init__(self, gpio_pin=17, pulses_per_rev=1, debounce_us=5000):
        super().__init__()

        self.TACH_PIN = gpio_pin              # BCM GPIO number
        self.PULSES_PER_REV = pulses_per_rev
        self.DEBOUNCE_US = debounce_us

        self.first_pulse = False
        self.last_tick = None

        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("❌ pigpio daemon not running (run: sudo pigpiod)")

        print(f"✅ Tachometer initialized on GPIO {self.TACH_PIN}")

        self.pi.set_mode(self.TACH_PIN, pigpio.INPUT)
        self.pi.set_pull_up_down(self.TACH_PIN, pigpio.PUD_UP)
        self.cb = self.pi.callback(self.TACH_PIN, pigpio.FALLING_EDGE, self._pulse_callback)

    def _pulse_callback(self, gpio, level, tick):
        print(f"🔥 CALLBACK FIRED | gpio={gpio} level={level} tick={tick} lasttick = {self.last_tick}")
        
        if level != 0:
            return

        if self.first_pulse is False:
            self.first_pulse = True
            self.first_pulse_detected.emit()

        if self.last_tick is not None:
            delta_us = pigpio.tickDiff(self.last_tick, tick)
            print(f"⏱️ Time since last pulse: {delta_us} µs")
        

            # Debounce / noise rejection
            if delta_us < self.DEBOUNCE_US:     
                print("⚠️ Pulse ignored (debounce)")
                return

            # ---- RPM CALCULATION (CORRECT) ----
            rpm = (60_000_000.0 / delta_us) / self.PULSES_PER_REV

            print(f"🟡 RPM calculated = {rpm:.2f}")
            self.rpm_updated.emit(rpm)

        self.last_tick = tick
        



    def cleanup(self):
        if self.cb:
            self.cb.cancel()
        if self.pi:
            self.pi.stop()
        print("🧹 Tachometer cleaned up")
