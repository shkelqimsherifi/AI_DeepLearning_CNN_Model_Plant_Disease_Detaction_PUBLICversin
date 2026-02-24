import sys
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QLabel)
from PySide6.QtGui import QPixmap, QImage, QFont, QColor
from PySide6.QtCore import Qt, QSize
import cv2

# Fluent Widgets
from qfluentwidgets import (PushButton, PrimaryPushButton, TitleLabel, BodyLabel, 
                            CaptionLabel, MessageBox, MSFluentWindow, Theme, 
                            setTheme, ThemeColor, setThemeColor, CardWidget,
                            ImageLabel)
from qfluentwidgets import FluentIcon as FIF

from model_handler import ModelHandler
from treatments_en import treatments_en
from treatments_sq import treatments_sq

UI_LABELS = {
    "en": {
        "title": "Plant Disease Recognition",
        "load_model": "Load Model",
        "select_image": "Select Image",
        "predict": "Predict",
        "original_image": "Original Image",
        "prediction_result": "Prediction Result",
        "no_image": "No Image Selected",
        "status_ready": "Status: Ready",
        "status_loading": "Status: Model loaded from {filename}",
        "status_selected": "Status: Image selected: {filename}",
        "status_predicting": "Status: Predicting...",
        "status_complete": "Status: Prediction Complete",
        "status_error": "Status: Error during prediction.",
        "lang_btn": "Shqip",
        "treatment_title": "Treatment Information",
        "error_title": "Error"
    },
    "sq": {
        "title": "Identifikimi i Sëmundjeve",
        "load_model": "Ngarko Modelin",
        "select_image": "Zgjidh Imazhin",
        "predict": "Parashiko",
        "original_image": "Imazhi Origjinal",
        "prediction_result": "Rezultati",
        "no_image": "Asnjë imazh",
        "status_ready": "Statusi: Gati",
        "status_loading": "Statusi: Modeli u ngarkua nga {filename}",
        "status_selected": "Statusi: Imazhi u zgjodh: {filename}",
        "status_predicting": "Statusi: Duke parashikuar...",
        "status_complete": "Statusi: Përfundoi",
        "status_error": "Statusi: Gabim gjatë parashikimit.",
        "lang_btn": "English",
        "treatment_title": "Informacioni i Trajtimit",
        "error_title": "Gabim"
    }
}

class MainWindow(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plant Disease Detection")
        self.setMinimumSize(1000, 750)
        
        # Set Theme
        setTheme(Theme.LIGHT)
        setThemeColor("#1a73e8")

        self.model_handler = ModelHandler()
        self.selected_image_path = None
        self.current_lang = "en"

        self.init_ui()

    def init_ui(self):
        # Create a central widget for the content
        self.container = QWidget()
        self.container.setObjectName("homeInterface")
        self.main_layout = QVBoxLayout(self.container)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(25)

        # Header
        self.header_label = TitleLabel(UI_LABELS[self.current_lang]["title"])
        self.header_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.header_label)

        # Controls Layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        
        self.load_model_btn = PushButton(FIF.FOLDER, UI_LABELS[self.current_lang]["load_model"])
        self.load_model_btn.setFixedWidth(160)
        self.load_model_btn.clicked.connect(self.on_load_model)

        self.select_img_btn = PushButton(FIF.PHOTO, UI_LABELS[self.current_lang]["select_image"])
        self.select_img_btn.setFixedWidth(160)
        self.select_img_btn.clicked.connect(self.on_select_image)

        self.predict_btn = PrimaryPushButton(FIF.PLAY, UI_LABELS[self.current_lang]["predict"])
        self.predict_btn.setFixedWidth(160)
        self.predict_btn.clicked.connect(self.on_predict)
        self.predict_btn.setEnabled(False)

        self.lang_btn = PushButton(FIF.LANGUAGE, UI_LABELS[self.current_lang]["lang_btn"])
        self.lang_btn.setFixedWidth(130)
        self.lang_btn.clicked.connect(self.toggle_language)

        controls_layout.addStretch(1)
        controls_layout.addWidget(self.load_model_btn)
        controls_layout.addWidget(self.select_img_btn)
        controls_layout.addWidget(self.predict_btn)
        controls_layout.addWidget(self.lang_btn)
        controls_layout.addStretch(1)
        self.main_layout.addLayout(controls_layout)

        # Content Area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left Card: Image Selection
        self.original_card = CardWidget()
        left_layout = QVBoxLayout(self.original_card)
        left_layout.setContentsMargins(15, 15, 15, 15)
        self.original_title = BodyLabel(UI_LABELS[self.current_lang]["original_image"])
        self.original_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_title)

        self.original_img_view = ImageLabel()
        self.original_img_view.setFixedSize(450, 450)
        self.original_img_view.setBorderRadius(8, 8, 8, 8)
        self.original_img_view.setText(UI_LABELS[self.current_lang]["no_image"])
        self.original_img_view.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.original_img_view)
        content_layout.addWidget(self.original_card)

        # Right Card: Prediction Results
        self.result_card = CardWidget()
        right_layout = QVBoxLayout(self.result_card)
        right_layout.setContentsMargins(15, 15, 15, 15)
        self.result_title = BodyLabel(UI_LABELS[self.current_lang]["prediction_result"])
        self.result_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.result_title)

        self.predicted_img_view = ImageLabel()
        self.predicted_img_view.setFixedSize(450, 450)
        self.predicted_img_view.setBorderRadius(8, 8, 8, 8)
        self.predicted_img_view.setText(UI_LABELS[self.current_lang]["no_image"])
        self.predicted_img_view.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.predicted_img_view)

        self.result_label = BodyLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a73e8;")
        right_layout.addWidget(self.result_label)
        
        content_layout.addWidget(self.result_card)
        self.main_layout.addLayout(content_layout)

        # Status Footer
        self.status_label = CaptionLabel(UI_LABELS[self.current_lang]["status_ready"])
        self.main_layout.addWidget(self.status_label)

        # Add central widget to FluentWindow
        self.addSubInterface(self.container, FIF.HOME, "Home")

    def toggle_language(self):
        self.current_lang = "sq" if self.current_lang == "en" else "en"
        self.update_ui_text()

    def update_ui_text(self):
        labels = UI_LABELS[self.current_lang]
        self.header_label.setText(labels["title"])
        self.load_model_btn.setText(labels["load_model"])
        self.select_img_btn.setText(labels["select_image"])
        self.predict_btn.setText(labels["predict"])
        self.lang_btn.setText(labels["lang_btn"])
        self.original_title.setText(labels["original_image"])
        self.result_title.setText(labels["prediction_result"])
        
        if not self.selected_image_path:
            self.original_img_view.setText(labels["no_image"])
            self.predicted_img_view.setText(labels["no_image"])

        # Update status label
        curr_status = self.status_label.text()
        if "Ready" in curr_status or "Gati" in curr_status:
            self.status_label.setText(labels["status_ready"])
        elif "Complete" in curr_status or "Përfundoi" in curr_status:
            self.status_label.setText(labels["status_complete"])

    def on_load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Keras Model (*.keras *.h5)")
        if file_path:
            success, message = self.model_handler.load_model(file_path)
            if success:
                self.status_label.setText(UI_LABELS[self.current_lang]["status_loading"].format(filename=os.path.basename(file_path)))
                if self.selected_image_path:
                    self.predict_btn.setEnabled(True)
            else:
                MessageBox("Error", f"Failed to load model: {message}", self).exec()

    def on_select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Images (*.png *.jpg *.jpeg *.JPG)")
        if file_path:
            self.selected_image_path = file_path
            self.original_img_view.setImage(file_path)
            self.status_label.setText(UI_LABELS[self.current_lang]["status_selected"].format(filename=os.path.basename(file_path)))
            
            if self.model_handler.model:
                self.predict_btn.setEnabled(True)

    def on_predict(self):
        if not self.selected_image_path:
            return

        self.status_label.setText(UI_LABELS[self.current_lang]["status_predicting"])
        result, error = self.model_handler.predict(self.selected_image_path)
        
        if error:
            MessageBox(UI_LABELS[self.current_lang]["error_title"], error, self).exec()
            self.status_label.setText(UI_LABELS[self.current_lang]["status_error"])
            return

        # Display result text
        label = result['label'].replace('___', ' - ')
        confidence = result['confidence'] * 100
        self.result_label.setText(f"{label}\n({confidence:.2f}%)")

        self.predicted_img_view.setImage(self.selected_image_path)
        self.status_label.setText(UI_LABELS[self.current_lang]["status_complete"])

        # Show Treatment Notification
        treatments = treatments_en if self.current_lang == "en" else treatments_sq
        original_label = result['label']
        treatment_text = treatments.get(original_label, "No treatment info available." if self.current_lang == "en" else "Nuk ka informacion për trajtimin.")
        
        # Modern Dialog for treatment
        title = UI_LABELS[self.current_lang]["treatment_title"]
        msg = MessageBox(title, treatment_text, self)
        msg.yesButton.setText("OK")
        msg.cancelButton.hide()
        msg.exec()

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
