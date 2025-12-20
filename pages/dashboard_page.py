# pages/dashboard_page.py

from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt

class DashboardPage(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(20)

        title_label = QLabel("Analysis Groups")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        title_label.setStyleSheet("""
            font-size: 2rem;
            font-weight: bold;
            color: #222;
        """)
        main_layout.addWidget(title_label)

        button_texts = [
            "Waveform and Spectrum",
            "Demodulation Spectrum",
            "Bump Test",
            "CoastDown/Run Up",
            "Rotor Balancing"
        ]

        for idx, text in enumerate(button_texts):
            button = QPushButton(text)
            button.setMinimumHeight(50)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setStyleSheet("""
                background-color: #007bff;
                color: white;
                font-size: 1.2rem;
                font-weight: bold;
                border-radius: 10px;
            """)
            button.clicked.connect(lambda checked, index=idx: self.handle_button_click(index))
            main_layout.addWidget(button)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def handle_button_click(self, index: int) -> None:
        if index == 0:
            self.main_window.stacked_widget.setCurrentIndex(1)
        if index == 4:
            self.main_window.stacked_widget.setCurrentIndex(2)  # Switch to Rotor Balancing Page
