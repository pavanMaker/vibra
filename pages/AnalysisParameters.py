from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QComboBox, QLabel,
    QGridLayout, QPushButton, QHBoxLayout, QDialog
)
from PyQt6.QtCore import Qt

class AnalysisParameter(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter List")
        self.setModal(True)
        self.setFixedSize(500, 500)

        layout = QVBoxLayout()

        heading = QLabel("Parameter List")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(heading)

        self.combo_measurement = QComboBox()
        self.combo_measurement.addItems(["Acceleration", "Velocity", "Displacement"])
        self.combo_Fmax = QComboBox()
        self.combo_Fmax.addItems(["144.0Hz","180.0Hz","225.0Hz","288.0Hz","360.0Hz","450.0Hz","576.0Hz","720.0Hz","900.0Hz","1.15kHz","1.44kHz","1.80kHz","2.30kHz","2.88kHz","3.60kHz","4.61kHz","5.76kHz","7.20kHz","9.22kHz","11.52kHz","14.40kHz","18.43kHz","23.04kHz","28.80kHz","36.86kHz","46.08kHz"])
        self.combo_No_of_Samples = QComboBox()
        self.combo_No_of_Samples.addItems([ "512/225", "1024/450", "2048/900", "4096/1800", "8192/3600", "16384/7200"])
        self.combo_Window_Type = QComboBox()
        self.combo_Window_Type.addItems(["Hanning", "Flat Top","Uniform"])

        fields = [
            ("Measurement Quantity", self.combo_measurement),
            ("Fmax(Hz)", self.combo_Fmax),
            ("Fmin(Hz) (0-95% Fmax)", QLineEdit()),
            ("Number of Samples/Lines", self.combo_No_of_Samples),
            ("Average Type", QLineEdit()),
            ("Average Number", QLineEdit()),
            ("Window Type", self.combo_Window_Type),
            ("Overlap Rate", QLineEdit()),
            ("Expected RPM", QLineEdit())
        ]
        

        form_grid = QGridLayout()
        for i, (label_text, widget) in enumerate(fields):
            label = QLabel(label_text)
            form_grid.addWidget(label, i, 0)
            form_grid.addWidget(widget, i, 1)
        layout.addLayout(form_grid)

        placeholder_box = QHBoxLayout()
        for _ in range(4):
            box = QLineEdit()
            box.setPlaceholderText("")
            box.setReadOnly(True)
            placeholder_box.addWidget(box)
        layout.addLayout(placeholder_box)

        button_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        apply_btn = QPushButton("Apply")
        cancel_btn.clicked.connect(self.reject)
        apply_btn.clicked.connect(self.apply_settings)

        button_row.addWidget(cancel_btn)
        button_row.addWidget(apply_btn)
        layout.addLayout(button_row)

        self.setLayout(layout)

    def apply_settings(self):
        self.parent().selected_quantity = self.combo_measurement.currentText()
        fmax_text = self.combo_Fmax.currentText().strip()

        # Normalize and convert
        try:
            if 'kHz' in fmax_text.lower():
                fmax_numeric = fmax_text.lower().replace('khz', '').strip()
                fmax_hz = float(fmax_numeric) * 1000
            elif 'hz' in fmax_text.lower():
                fmax_numeric = fmax_text.lower().replace('hz', '').strip()
                fmax_hz = float(fmax_numeric)
            else:
                # fallback
                fmax_hz = float(fmax_text.strip())
            
            self.parent().selected_fmax_hz = fmax_hz
            print(f"✅ Fmax set to: {fmax_hz} Hz")

        except ValueError:
            print(f"❌ Invalid Fmax format: {fmax_text}")
            self.parent().selected_fmax_hz = 500.0  # fallback value

        self.close()
