import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎭 Real-Time Emotion Detection")
        self.setFixedSize(900, 600)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)

        self.result_label = QLabel("Detected Emotion: None")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; padding: 10px;")

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        # Layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_label)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Timer for video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Webcam
        self.cap = None

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.result_label.setText("Detected Emotion: None")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion_text = "None"
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)
            emotion_text = emotion_labels[np.argmax(preds)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        self.result_label.setText(f"Detected Emotion: {emotion_text}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qImg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec())
