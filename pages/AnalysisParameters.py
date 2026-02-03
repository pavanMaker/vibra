from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QComboBox, QLabel,
    QGridLayout, QPushButton, QHBoxLayout, QDialog,QTableWidget,QFormLayout,QTableWidgetItem
)
from PyQt6.QtCore import Qt
from pages.settings_manager import save_settings
from backend.mcc_backend import Mcc172Backend
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
        self.combo_Fmax.addItems(["144.0Hz","180.0Hz","225.0Hz","288.0Hz","360.0Hz","450.0Hz","576.0Hz","720.0Hz","900.0Hz","1150Hz","1440Hz","1.80kHz","2.30kHz","2.88kHz","3.60kHz","4.61kHz","5.76kHz","7.20kHz","9.22kHz","11.52kHz","14.40kHz","18.43kHz","23.04kHz","28.80kHz","36.86kHz","46.08kHz"])
        self.combo_Fmax.setStyleSheet(font_style)

        self.fmin_input = QLineEdit()
        self.fmin_input.setReadOnly(True)
        self.fmin_input.setStyleSheet(font_style)
        self.fmin_input.mousePressEvent = self.create_keypad_layout 

        self.combo_No_of_Samples = QComboBox()
        self.combo_No_of_Samples.addItems([ "512/225", "1024/450", "2048/900", "4096/1800", "8192/3600", "16384/7200","65536/25600"])
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
        self.restore_previous_settings()
    def restore_previous_settings(self):
        parent = self.parent()
        def safe_set_combo(combo: QComboBox, value):
            if value is None:
                combo.setCurrentIndex(0)
                return
            
            idx = combo.findText(str(value), Qt.MatchFlag.MatchFixedString)
            if idx != -1:
                combo.setCurrentIndex(idx)
            else:
                try:
                    fv = float(value)

                    found = False
                    for i in range(combo.count()):
                        item = combo.itemText(i).lower()

                        digits = ''.join(c for c in item if c.isdigit() or c == '.')
                        if not digits:
                            continue
                        v = float(digits)
                        if "khz" in item:
                            v *= 1000.0
                        if abs(v - fv) < 1e-6 or abs(v - fv) / max(1.0, fv) < 1e-6:
                            combo.setCurrentIndex(i)
                            found =  True
                            break
                        if not found:
                            combo.setCurrentIndex(0)
                except Exception:
                    combo.setCurrentIndex(0)


 

        # 1. Measurement Quantity
        safe_set_combo(self.combo_measurement, getattr(parent, 'selected_quantity', "Acceleration"))

        fmax_val = getattr(parent, 'selected_fmax_hz', None)
        if fmax_val is not None:
            safe_set_combo(self.combo_Fmax, fmax_val)
        # 2. Fmax: you can add a robust Fmax restore if needed
        # 3. Fmin
        if hasattr(parent, 'selected_fmin_hz') and parent.selected_fmin_hz is not None:
            try:
                self.fmin_input.setText(str(float(parent.selected_fmin_hz)))
            except Exception:
                self.fmin_input.setText(str(parent.selected_fmin_hz))
            
        # 4. Number of Samples/Lines
        buffer_size = getattr(parent, 'buffer_size', None)
        if buffer_size is None:
            buffer_size = getattr(getattr(parent, 'daq', None), 'buffer_size', None)
        if buffer_size is None:
            buffer_size = 4096

        buffer_size_str = str(int(buffer_size))
        

        found = False
        for i in range(self.combo_No_of_Samples.count()):
            left = self.combo_No_of_Samples.itemText(i).split("/")[0].strip()
            if left == buffer_size_str:
                self.combo_No_of_Samples.setCurrentIndex(i)
                found = True
                break
        if not found:
            self.combo_No_of_Samples.setCurrentIndex(self.combo_No_of_Samples.count() - 1)
        # 5. Window Type
        if hasattr(parent, 'selected_window_type'):
            safe_set_combo(self.combo_Window_Type, parent.selected_window_type)

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
        parent = self.parent()
        parent.selected_quantity = self.combo_measurement.currentText()
        fmax_text = self.combo_Fmax.currentText().strip().lower()
        fmax_hz = None
        try:
            if 'khz' in fmax_text:
                clean_value = ''.join(c for c in fmax_text if c.isdigit() or c == '.')
                fmax_hz = float(clean_value) * 1000
                sampling_frequency = fmax_hz * 2.56
                print("Setting sampling frequency to:", sampling_frequency)
            elif 'hz' in fmax_text:
                clean_value = ''.join(c for c in fmax_text if c.isdigit() or c == '.')
                fmax_hz = float(clean_value)
                sampling_frequency = fmax_hz * 2.56
            else:
                fmax_hz = float(fmax_text)
            parent.selected_fmax_hz = fmax_hz
            sampling_frequency = fmax_hz * 2.56
            parent.daq.sample_rate = sampling_frequency
            print(" Fmax set to:", parent.selected_fmax_hz, "Hz")
        except Exception:
            print(f" Invalid Fmax format: {fmax_text}")
            parent.selected_fmax_hz = 500.0  # fallback

        fmin_txt = self.fmin_input.text().strip()
        if fmin_txt:
            try:
                parent.selected_fmin_hz = float(fmin_txt)
            except Exception:
                print("❌ Invalid Fmin format:")


        selected_samples_text = self.combo_No_of_Samples.currentText()
        try:
            selected_samples = int(selected_samples_text.split("/")[0])
            parent.buffer_size = selected_samples
            daq = getattr(parent, 'daq', None)
            if daq is not None:
                daq.buffer_size = selected_samples
                try:
                    daq.setup()
                except Exception as e:
                    print("? Error setting up DAQ:", e)

                
            
            
            print("✅ Number of Samples set to:", selected_samples)
        except Exception:
            print(f"❌ Error setting buffer size on DAQ backend.",selected_samples)
            parent.buffer_size = getattr(parent,"buffer_size",8192)

         # fallback value

            

        to_save = {
            "selected_quantity": parent.selected_quantity,
            "selected_fmax_hz":  (parent.selected_fmax_hz),
            "selected_fmin_hz":  float(getattr(parent, "selected_fmin_hz", 1.0)),
            "top_channel": int(getattr(parent, "top_channel", 0)),
            "bottom_channel": int(getattr(parent, "bottom_channel", 1)),
            "trace_mode_index": int(getattr(parent, "stacked_views", None).currentIndex() if getattr(parent, "stacked_views", None) else 0),
            "display_settings": getattr(parent, "display_settings", {}),
            "buffer_size": int(getattr(parent, "buffer_size", 4096)),
            "input_channels": getattr(parent, "input_channels", None).data if getattr(parent, "input_channels", None) else None,
            "sample_rate": parent.daq.sample_rate
        }
        save_settings(to_save)

        self.accept()




# input_channels.py  (patched)


class InputChannelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Channels")
        self.setMinimumSize(600, 300)

        self.channels = 6
        self.data = [
            dict(channel=f"Ch{i}",
                 quantity="Acceleration",
                 sensitivity="100",
                 unit="mv/g",
                 input_mode="IEPE")
            for i in range(self.channels)
        ]

        layout = QVBoxLayout(self)

        self.table = QTableWidget(self.channels, 3)
        self.table.setHorizontalHeaderLabels(["Channel", "Sensitivity", "Input Mode"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.populate_table()

        self.table.cellClicked.connect(self.show_sensor_dialog)

        layout.addWidget(self.table)

        btns = QHBoxLayout()
        btns.addWidget(QPushButton("Apply", clicked=self.accept))
        btns.addWidget(QPushButton("Close", clicked=self.reject))
        layout.addLayout(btns)

    def populate_table(self):
        for row, ch in enumerate(self.data):
            self.table.setItem(row, 0, QTableWidgetItem(ch["channel"]))

            txt = f'{ch["quantity"]} {ch["sensitivity"]} {ch["unit"]}'
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, item)

            combo = QComboBox()
            combo.addItems(["IEPE", "Single Ended"])
            combo.setCurrentText(ch["input_mode"])
            combo.currentTextChanged.connect(lambda v, r=row: self._set_mode(r, v))
            self.table.setCellWidget(row, 2, combo)

    def _set_mode(self, row: int, mode: str):
        self.data[row]["input_mode"] = mode

    def show_sensor_dialog(self, row: int, col: int):
        if col != 1:
            return

        d = QDialog(self)
        d.setWindowTitle(f"Sensor Settings – {self.data[row]['channel']}")
        d.setMinimumWidth(400)
        form = QFormLayout(d)

        quantity = QComboBox()
        quantity.addItems(["Acceleration", "Velocity", "Displacement"])
        quantity.setCurrentText(self.data[row]["quantity"])

        unit = QComboBox()
        sens = QLineEdit(self.data[row]["sensitivity"])
        sens.setPlaceholderText("Enter number only")

        def refresh_units():
            unit_options = {
                "Acceleration": ["g", "m/s²", "mm/s²", "in/s²"],
                "Velocity":     ["m/s", "cm/s", "in/s"],
                "Displacement": ["m", "mm", "µm", "in"]
            }
            selected = quantity.currentText()
            unit.clear()
            unit.addItems(unit_options[selected])
            current_unit = self.data[row]["unit"]
            if current_unit in unit_options[selected]:
                unit.setCurrentText(current_unit)

        quantity.currentTextChanged.connect(refresh_units)
        refresh_units()

        form.addRow("Measurement Quantity", quantity)
        form.addRow("Sensor Engineering Unit", unit)
        form.addRow("Sensor Sensitivity (numeric)", sens)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(cancel_btn)
        form.addRow(btn_row)

        def apply_values():
            q = quantity.currentText()
            u = unit.currentText()
            val = sens.text().strip()
            try:
                float(val)  # Ensure it's numeric
            except ValueError:
                sens.setStyleSheet("border: 2px solid red")
                return

            self.data[row]["quantity"] = q
            self.data[row]["unit"] = u
            self.data[row]["sensitivity"] = val
            self.table.item(row, 1).setText(f"{q} {val} mV/{u}")
            d.accept()

        apply_btn.clicked.connect(apply_values)
        cancel_btn.clicked.connect(d.reject)
        d.exec()

# display_preferences.py



class DisplayPreferences(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Display Preferences")
        self.setMinimumSize(400, 400)

        layout = QVBoxLayout()

        title = QLabel("Display Preferences")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self.settings = {}

        # Helper to add each dropdown group
        def add_group(label_text, options):
            layout.addWidget(QLabel(label_text))
            combo = QComboBox()
            combo.addItems(options)
            current =  self.parent().display_settings.get(label_text,options[0])
            combo.setCurrentText(current)
            combo.setStyleSheet("font-size: 16px; padding: 4px;")
            layout.addWidget(combo)
            self.settings[label_text] = combo

        # Acceleration
        add_group("Acceleration Spectrum Type", ["RMS","Peak", "Peak-Peak"])
        add_group("Acceleration Engineering Unit", ["g", "m/s²", "mm/s²", "cm/s²", "in/s²"])

        # Velocity
        add_group("Velocity Spectrum Type", ["RMS", "Peak", "Peak-Peak"])
        add_group("Velocity Engineering Unit", ["m/s", "cm/s", "mm/s", "in/s"])

        # Displacement
        add_group("Displacement Spectrum Type", ["RMS", "Peak", "Peak-Peak"])
        add_group("Displacement Engineering Unit", ["m", "cm", "mm", "in","μm"])

        # Buttons
        btns = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        apply_btn.setStyleSheet("font-size: 16px; padding: 6px 16px; font-weight: bold;")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        close_btn.setStyleSheet("font-size: 16px; padding: 6px 16px; font-weight: bold;")
        btns.addWidget(apply_btn)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

        self.setLayout(layout)

    def apply_settings(self):
        for label, combo in self.settings.items():
            self.parent().display_settings[label] = combo.currentText()
            print("units :",self.parent().display_settings[label])
        save_settings({

            "selected_quantity": self.parent().selected_quantity,
            "selected_fmax_hz": self.parent().selected_fmax_hz,
            "selected_fmin_hz": self.parent().selected_fmin_hz,
            "top_channel": self.parent().top_channel,
            "bottom_channel": self.parent().bottom_channel,
            "trace_mode_index": self.parent().stacked_views.currentIndex(),
            "display_settings": self.parent().display_settings
        })
            

        self.accept()
        

    def closeEvent(self, event):
        """Ensure proper cleanup on X close"""
        self.reject()
        event.accept()