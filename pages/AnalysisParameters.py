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
        self.combo_No_of_Samples = QComboBox()
        self.combo_No_of_Samples.addItems([ "512/225", "1024/450", "2048/900", "4096/1800", "8192/3600", "16384/7200"])
        self.combo_Window_Type = QComboBox()
        self.combo_Window_Type.addItems(["Hanning", "Flat Top","Uniform"])
        self.combo_Fmax = QComboBox()
        self.combo_Fmax.addItems(["3.60kHz", "2.88kHz", "2.30kHz", "1.44kHz", "1.15kHz", "900.0Hz", "720.0Hz", "576.0Hz", "450.0Hz"])

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
        fmax_text = self.combo_Fmax.currentText()
        fmax_hz = float(fmax_text.replace("kHz", "").replace("Hz", "")) * (1000 if "kHz" in fmax_text else 1)
        self.parent().selected_fmax = fmax_hz
        self.close()