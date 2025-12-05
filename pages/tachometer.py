# tachometer.py
import pigpio
from PyQt6.QtCore import QObject, pyqtSignal

class TachometerReader(QObject):
    rpm_updated = pyqtSignal(float)  # Signal to emit RPM to GUI

    def __init__(self, gpio_pin=17, pulses_per_rev=1, debounce_us=5000):
        super().__init__()
        self.TACH_PIN = gpio_pin
        self.PULSES_PER_REV = pulses_per_rev
        self.DEBOUNCE_MICROSECONDS = debounce_us
        self.last_tick = None
        self.pi = pigpio.pi()

        if not self.pi.connected:
            raise RuntimeError("❌ pigpio daemon not running. Start it with: sudo pigpiod")

        self.pi.set_mode(self.TACH_PIN, pigpio.INPUT)
        self.pi.set_pull_up_down(self.TACH_PIN, pigpio.PUD_UP)
        self.cb = self.pi.callback(self.TACH_PIN, pigpio.FALLING_EDGE, self._pulse_callback)

    def _pulse_callback(self, gpio, level, tick):
        if level != 0:
            return

        if self.last_tick is not None:
            delta_us = pigpio.tickDiff(self.last_tick, tick)
            if delta_us >= self.DEBOUNCE_MICROSECONDS:
                rpm = (60_000_000 / delta_us) / self.PULSES_PER_REV
                self.rpm_updated.emit(rpm)
        self.last_tick = tick

    def cleanup(self):
        if self.cb:
            self.cb.cancel()
        if self.pi:
            self.pi.stop()