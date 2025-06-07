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
        self.setFixedSize(600, 600)

        layout = QVBoxLayout()

        heading = QLabel("Parameter List")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        layout.addWidget(heading)

        font_style = "font-size: 16px; padding: 4px;"
        self.combo_measurement = QComboBox()
        self.combo_measurement.addItems(["Acceleration", "Velocity", "Displacement"])
        self.combo_measurement.setStyleSheet(font_style)

        self.combo_Fmax = QComboBox()
        self.combo_Fmax.addItems(["144.0Hz","180.0Hz","225.0Hz","288.0Hz","360.0Hz","450.0Hz","576.0Hz","720.0Hz","900.0Hz","1.15kHz","1.44kHz","1.80kHz","2.30kHz","2.88kHz","3.60kHz","4.61kHz","5.76kHz","7.20kHz","9.22kHz","11.52kHz","14.40kHz","18.43kHz","23.04kHz","28.80kHz","36.86kHz","46.08kHz"])
        self.combo_Fmax.setStyleSheet(font_style)

        self.fmin_input = QLineEdit()
        self.fmin_input.setReadOnly(True)
        self.fmin_input.setStyleSheet(font_style)
        self.fmin_input.mousePressEvent = self.create_keypad_layout 

        self.combo_No_of_Samples = QComboBox()
        self.combo_No_of_Samples.addItems([ "512/225", "1024/450", "2048/900", "4096/1800", "8192/3600", "16384/7200"])
        self.combo_No_of_Samples.setStyleSheet(font_style)

        self.combo_Window_Type = QComboBox()
        self.combo_Window_Type.addItems(["Hanning", "Flat Top", "Uniform"])
        self.combo_Window_Type.setStyleSheet(font_style)

        fields = [
            ("Measurement Quantity", self.combo_measurement),
            ("Fmax(Hz)", self.combo_Fmax),
            ("Fmin(Hz) (0-95% Fmax)", self.fmin_input),
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
            label.setStyleSheet("font-size: 16px; font-weight: bold;")
            if isinstance(widget, QLineEdit):
                widget.setStyleSheet(font_style)
            form_grid.addWidget(label, i, 0)
            form_grid.addWidget(widget, i, 1)
        layout.addLayout(form_grid)

        button_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        apply_btn = QPushButton("Apply")
        for btn in [cancel_btn, apply_btn]:
            btn.setMinimumHeight(40)
            btn.setStyleSheet("""
                font-size: 16px;
                font-weight: bold;
                padding: 6px 20px;
                border-radius: 8px;
            """)
        cancel_btn.clicked.connect(self.reject)
        apply_btn.clicked.connect(self.apply_settings)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(apply_btn)
        layout.addLayout(button_row)

        self.setLayout(layout)

    def create_keypad_layout(self, event):
        popup = QDialog(self)
        popup.setWindowTitle("Enter Fmin (Hz)")
        popup.setFixedSize(350, 400)

        layout = QVBoxLayout(popup)
        input_field = QLineEdit()
        input_field.setReadOnly(True)
        input_field.setAlignment(Qt.AlignmentFlag.AlignRight)
        input_field.setStyleSheet("font-size: 18px; padding: 10px;")
        layout.addWidget(input_field)

        from PyQt6.QtWidgets import QGridLayout
        keypad = QGridLayout()
        keys = [
            ('1', 0, 0), ('2', 0, 1), ('3', 0, 2),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2),
            ('7', 2, 0), ('8', 2, 1), ('9', 2, 2),
            ('+/-', 3, 0), ('0', 3, 1), ('BKsp', 3, 2),
            ('Clear', 4, 0), ('Cancel', 4, 1), ('Apply', 4, 2)
        ]

        def handle_layout(key):
            current = input_field.text()
            if key == 'Clear':
                input_field.clear()
            elif key == 'BKsp':
                input_field.setText(current[:-1])
            elif key == '+/-':
                input_field.setText(current[1:] if current.startswith('-') else '-' + current)
            elif key == 'Cancel':
                popup.reject()
            elif key == 'Apply':
                try:
                    fmin_val = float(input_field.text())
                    self.fmin_input.setText(str(fmin_val))
                    self.parent().selected_fmin_hz = fmin_val
                    popup.accept()
                except ValueError:
                    popup.reject()
            else:
                input_field.setText(current + key)

        for text, row, col in keys:
            btn = QPushButton(text)
            btn.setStyleSheet("font-size: 16px; padding: 10px;")
            btn.setMinimumSize(80, 40)
            btn.clicked.connect(lambda _, t=text: handle_layout(t))
            keypad.addWidget(btn, row, col)

        layout.addLayout(keypad)
        popup.exec()

    def apply_settings(self):
        self.parent().selected_quantity = self.combo_measurement.currentText()
        fmax_text = self.combo_Fmax.currentText().strip()
        try:
            if 'kHz' in fmax_text.lower():
                fmax_hz = float(fmax_text.lower().replace('khz', '').strip()) * 1000
            elif 'hz' in fmax_text.lower():
                fmax_hz = float(fmax_text.lower().replace('hz', '').strip())
            else:
                fmax_hz = float(fmax_text.strip())
            self.parent().selected_fmax_hz = fmax_hz
            print(f"✅ Fmax set to: {fmax_hz} Hz")
        except ValueError:
            print(f"❌ Invalid Fmax format: {fmax_text}")
            self.parent().selected_fmax_hz = 500.0  # fallback value

        self.close()
