import cv2
import torch
import os
import time
import threading
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk

# ================================
# CONFIGURATION SETTINGS
# ================================
VIDEO_SOURCE = 0  # Change to 1 or 2 if you have multiple cameras, or a file path like 'video.mp4'
CONFIDENCE_THRESHOLD = 0.4
OUTPUT_DIR = 'output'
LOG_FILE = os.path.join(OUTPUT_DIR, 'detection_log.txt')
SAVE_VIDEO = True
SOUND_ALERT = True

ANIMAL_CLASSES = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe','deer', 'fox', 'rabbit', 'wolf', 'monkey', 'panda',
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class AnimalDetectionApp:
    def __init__(self):   # ✅ FIXED: double underscores
        self.root = ctk.CTk()
        self.setup_ui()
        self.model = None
        self.stop_detection_flag = False
        self.detection_active = False
        self.current_frame = None
        self.cap = None
        self.out = None
        
    def setup_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.root.title("WildEye Animal Detector")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header_frame = ctk.CTkFrame(self.main_frame, height=80, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="WildEye Animal Detector",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#4e89e8"
        )
        self.title_label.pack(side="left", padx=(20, 0))
        
        # Status indicator
        self.status_label = ctk.CTkLabel(
            self.header_frame,
            text="■ IDLE",
            text_color="gray",
            font=ctk.CTkFont(size=14),
            fg_color="#252525",
            corner_radius=10,
            height=30,
            padx=15
        )
        self.status_label.pack(side="right", padx=(0, 20))
        
        # Main content area
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.content_frame.grid_columnconfigure(0, weight=3)
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Video panel
        self.video_frame = ctk.CTkFrame(self.content_frame)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        
        self.canvas = ctk.CTkCanvas(self.video_frame, bg="#252525", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.fps_label = ctk.CTkLabel(
            self.video_frame,
            text="FPS: 0.00",
            text_color="white",
            font=ctk.CTkFont(size=12)
        )
        self.fps_label.pack(side="bottom", anchor="e", padx=10, pady=5)
        
        # Control panel
        self.control_frame = ctk.CTkFrame(self.content_frame, width=300)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)
        self.control_frame.grid_propagate(False)
        
        # Settings group
        settings_group = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        settings_group.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            settings_group,
            text="Detection Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w")
        
        # Confidence threshold slider
        ctk.CTkLabel(
            settings_group,
            text="Confidence Threshold:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(10, 0))
        
        self.confidence_slider = ctk.CTkSlider(
            settings_group,
            from_=0.1,
            to=0.9,
            number_of_steps=8,
            command=self.update_confidence_threshold
        )
        self.confidence_slider.set(CONFIDENCE_THRESHOLD)
        self.confidence_slider.pack(fill="x", pady=(0, 10))
        
        self.confidence_value = ctk.CTkLabel(
            settings_group,
            text=f"{CONFIDENCE_THRESHOLD*100:.0f}%",
            font=ctk.CTkFont(size=12)
        )
        self.confidence_value.pack(anchor="e")
        
        # Controls
        ctk.CTkLabel(
            settings_group,
            text="Detection Controls",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(20, 0))
        
        self.start_button = ctk.CTkButton(
            settings_group,
            text="Start Detection",
            fg_color="#4e89e8",
            hover_color="#3a6fc4",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.start_detection
        )
        self.start_button.pack(fill="x", pady=10)
        
        self.stop_button = ctk.CTkButton(
            settings_group,
            text="Stop Detection",
            fg_color="#e84e4e",
            hover_color="#c43a3a",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            state="disabled",
            command=self.stop_detection
        )
        self.stop_button.pack(fill="x", pady=(0, 10))
        
        # Log panel
        log_group = ctk.CTkFrame(self.control_frame)
        log_group.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        ctk.CTkLabel(
            log_group,
            text="Detection Log",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        self.log_text = ctk.CTkTextbox(
            log_group,
            font=ctk.CTkFont(size=12),
            wrap="word",
            height=150
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_message("Application initialized and ready")
        
        # Footer
        self.footer_frame = ctk.CTkFrame(self.main_frame, height=30, fg_color="transparent")
        self.footer_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(
            self.footer_frame,
            text="WildEye Animal Detector v1.0 | © 2023",
            text_color="gray",
            font=ctk.CTkFont(size=12)
        ).pack(side="right", padx=10)
        
        # Bind ESC key to quit
        self.root.bind("<Escape>", lambda e: self.root.quit())
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        self.log_text.configure(state="normal")
        self.log_text.insert("0.0", timestamp + message + "\n")
        if float(self.log_text.index("end-1c").split('.')[0]) > 50:
            self.log_text.delete("end-1c linestart", "end-1c")
        self.log_text.configure(state="disabled")
        
    def update_confidence_threshold(self, value):
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = value
        self.confidence_value.configure(text=f"{value*100:.0f}%")
        
    def start_detection(self):
        self.detection_active = True
        self.stop_detection_flag = False
        self.status_label.configure(text="■ RUNNING", text_color="lightgreen")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.log_message("Starting animal detection...")
        
        # Initialize model if not loaded
        if self.model is None:
            self.log_message("Loading YOLOv5 model (first time may take longer)...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to("cpu")
            
        # Start detection thread
        threading.Thread(target=self.run_detection, daemon=True).start()
        
    def stop_detection(self):
        self.stop_detection_flag = True
        self.detection_active = False
        self.status_label.configure(text="■ STOPPED", text_color="orange")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.log_message("Stopping detection...")
        
    def run_detection(self):
        # Initialize video capture
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        time.sleep(1)  # Allow time for the camera to initialize
        if not self.cap.isOpened():
            self.log_message("Error: Could not open video source")
            messagebox.showerror("Error", "Could not open video source")
            return

        if SAVE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'output.avi'), 
                                     fourcc, 20.0, (640, 480))

        frame_count = 0
        animal_frame_count = 0
        start_time = time.time()

        while self.detection_active:
            ret, frame = self.cap.read()
            if not ret:
                self.log_message("Error: Failed to capture frame")
                break

            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform detection
            results = self.model(rgb_frame)
            detections = results.pandas().xyxy[0]
            
            # Filter detections to only include animals in ANIMAL_CLASSES
            animal_detections = detections[detections['name'].isin(ANIMAL_CLASSES)]

            animal_detected = False
            for _, row in animal_detections.iterrows():
                label = row['name']
                confidence = float(row['confidence'])
                if confidence > CONFIDENCE_THRESHOLD:
                    animal_detected = True
                    x1, y1 = int(row['xmin']), int(row['ymin'])
                    x2, y2 = int(row['xmax']), int(row['ymax'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} {confidence:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.log_detection(label, confidence, timestamp)
                    
                    if animal_frame_count == 0:  # Log first detection only
                        self.log_message(f"Detected {label} ({confidence:.2f})")

            if animal_detected:
                animal_frame_count += 1
                self.draw_alert(frame)
                if SOUND_ALERT:
                    threading.Thread(target=self.play_sound).start()

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Update UI
            self.update_frame(frame, fps)
            self.root.update()
            
            # Save video if enabled
            if SAVE_VIDEO and self.out:
                self.out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.out:
            self.out.release()
        
        if self.detection_active:  # If terminated unexpectedly
            self.stop_detection()
            
    def update_frame(self, frame, fps):
        self.fps_label.configure(text=f"FPS: {fps:.2f}")
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.current_frame = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.current_frame, anchor="nw")
        
    def log_detection(self, label, confidence, timestamp):
        with open(LOG_FILE, 'a') as f:
            f.write(f"{timestamp} - Detected: {label} ({confidence:.2f})\n")
            
    def draw_alert(self, frame):
        cv2.putText(frame, "ALERT: Animal Detected!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
    def play_sound(self):
        try:
            import winsound
            winsound.Beep(1000, 500)
        except ImportError:
            print('\a', end='')
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":   # ✅ FIXED: double underscores
    app = AnimalDetectionApp()
    app.run()
