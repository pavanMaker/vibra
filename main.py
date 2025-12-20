# main.py

from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from backend.mcc_backend import Mcc172Backend
from pages.dashboard_page import DashboardPage
from pages.Waveform import WaveformPage
from pages.rotor_balancing_page import RotorPage  # Import your updated PyQt6 WaveformPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vibration Analyzer")
        self.resize(900,700)
        

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create pages
        self.dashboard_page = DashboardPage(self)
        self.waveform_page = WaveformPage(self)
        self.rotor_page = RotorPage(self)  # Create an instance of RotorPage

        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.dashboard_page)  # index 0
        self.stacked_widget.addWidget(self.waveform_page)   # index 1
        self.stacked_widget.addWidget(self.rotor_page)      # index 2

        # Set initial page
        self.stacked_widget.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()  # ✅ In PyQt6, it is app.exec(), not app.exec_()
