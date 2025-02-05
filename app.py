import sys
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QStatusBar
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, Qt
from ultralytics import YOLO

class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.setGeometry(100, 100, 800, 600)
        
        # Load the trained model
        self.detection_model = YOLO("best_100.pt")  
        self.device = "cpu"  
        self.video_path = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.is_paused = True  # Start in paused state
        self.is_video_finished = False  # Track if the video has finished playing
        self.output_video_writer = None  # Video writer for saving the output
        self.is_saving = False  # Track if the user has clicked "Save"
        
        self.initialize_ui()
    
    def initialize_ui(self):
        layout = QVBoxLayout()
        
        # Video display
        self.video_display_label = QLabel(self)
        self.video_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display_label.setText("Video will be displayed here")
        self.video_display_label.setMinimumSize(640, 480)  # Set a minimum size for the video display
        layout.addWidget(self.video_display_label, stretch=1)  # Allow the video label to expand
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.upload_video_button = QPushButton("Select Video")
        self.upload_video_button.clicked.connect(self.load_video)
        button_layout.addWidget(self.upload_video_button)
        
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setEnabled(False)
        button_layout.addWidget(self.play_pause_button)
        
        self.save_video_button = QPushButton("Save Video")
        self.save_video_button.clicked.connect(self.save_video)
        self.save_video_button.setEnabled(False)
        button_layout.addWidget(self.save_video_button)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)
        
        self.setLayout(layout)
        
        # Set a minimum window size
        self.setMinimumSize(800, 600)
    
    def load_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi)")
        if file_path:
            self.video_path = file_path
            self.video_capture = cv2.VideoCapture(file_path)
            self.play_pause_button.setEnabled(True)
            self.save_video_button.setEnabled(True)
            self.status_bar.showMessage(f"Loaded video: {file_path}")
            self.is_video_finished = False  # Reset video finished flag
    
    def toggle_play_pause(self):
        if self.is_video_finished:
            # Restart the video if it has finished
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_video_finished = False
            self.is_paused = False
            self.timer.start(30)
            self.play_pause_button.setText("Pause")
            self.status_bar.showMessage("Video restarted.")
        elif self.is_paused:
            # Resume the video
            self.is_paused = False
            self.timer.start(30)
            self.play_pause_button.setText("Pause")
            self.status_bar.showMessage("Video resumed.")
        else:
            # Pause the video
            self.is_paused = True
            self.timer.stop()
            self.play_pause_button.setText("Play")
            self.status_bar.showMessage("Video paused.")
    
    def process_frame(self):
        if self.is_paused:
            return
        
        ret, frame = self.video_capture.read()
        if not ret:
            # Video has finished playing
            self.timer.stop()
            self.is_paused = True
            self.is_video_finished = True
            self.play_pause_button.setText("Play")
            self.status_bar.showMessage("Video finished.")
            if self.output_video_writer is not None:
                self.output_video_writer.release()
                self.output_video_writer = None
                self.is_saving = False
            return
        
        original_height, original_width = frame.shape[:2]
        
        # Resize frame to 640x640 for YOLO input
        resized_frame = cv2.resize(frame, (640, 640))
        
        # Perform inference
        detection_results = self.detection_model(resized_frame, device=self.device)
        
        for result in detection_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                label = result.names[class_id]
                
                # Scale bounding boxes back to original size
                x1 = int(x1 * original_width / 640)
                y1 = int(y1 * original_height / 640)
                x2 = int(x2 * original_width / 640)
                y2 = int(y2 * original_height / 640)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Write the frame to the output video file if saving is enabled
        if self.is_saving and self.output_video_writer is not None:
            self.output_video_writer.write(frame)
        
        # Convert the frame to QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale the QImage to fit the QLabel while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_display_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_display_label.setPixmap(scaled_pixmap)
    
    def save_video(self):
        if self.video_capture is not None and not self.is_saving:
            # Initialize the video writer
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            self.output_video_writer = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            self.is_saving = True
            self.status_bar.showMessage("Saving video...")
    
    def resizeEvent(self, event):
        # Resize the video display when the window is resized
        if self.video_display_label.pixmap():
            scaled_pixmap = self.video_display_label.pixmap().scaled(
                self.video_display_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.video_display_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        if self.output_video_writer is not None:
            self.output_video_writer.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())