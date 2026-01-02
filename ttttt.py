import cv2
import torch
import os
import time
import threading
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk

# Configuration Settings
VIDEO_SOURCE = 0  # Default camera (try 0, 1, 2, etc.)
CONFIDENCE_THRESHOLD = 0.4
OUTPUT_DIR = 'output'
LOG_FILE = os.path.join(OUTPUT_DIR, 'detection_log.txt')
SAVE_VIDEO = False  # Disabled for stability
SOUND_ALERT = True
ANIMAL_CLASSES = [
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class AnimalDetectionApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.setup_ui()
        self.model = None
        self.detection_active = False
        self.current_frame = None
        self.cap = None
        self.last_detection_time = 0
        self.frame_count = 0
        self.start_time = time.time()

    def setup_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.root.title("WildEye Animal Detector")
        self.root.geometry("1200x800")
        
        # Main layout with 3 rows: header, content, footer
        self.root.grid_rowconfigure(0, weight=0)  # Header
        self.root.grid_rowconfigure(1, weight=1)   # Content (expands)
        self.root.grid_rowconfigure(2, weight=0)   # Footer
        self.root.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_frame = ctk.CTkFrame(self.root)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="WildEye Animal Detector",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#4e89e8"
        )
        self.title_label.pack(side="left", padx=20)
        
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
        self.status_label.pack(side="right", padx=20)
        
        # Main content area (video + controls)
        self.content_frame = ctk.CTkFrame(self.root)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Configure content frame columns (80% video, 20% controls)
        self.content_frame.grid_columnconfigure(0, weight=4)
        self.content_frame.grid_columnconfigure(1, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Video display (left side)
        self.video_container = ctk.CTkFrame(self.content_frame)
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.canvas = ctk.CTkCanvas(self.video_container, bg="#252525")
        self.canvas.pack(fill="both", expand=True)
        
        # FPS counter
        self.fps_label = ctk.CTkLabel(
            self.video_container,
            text="FPS: 0.00",
            text_color="white",
            font=ctk.CTkFont(size=12)
        )
        self.fps_label.pack(side="bottom", anchor="e", padx=10, pady=5)
        
        # Control panel (right side)
        self.control_frame = ctk.CTkFrame(self.content_frame, width=300)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Confidence threshold control
        ctk.CTkLabel(
            self.control_frame,
            text="Confidence Threshold:",
            font=ctk.CTkFont(size=14)
        ).pack(pady=(10, 0))
        
        self.confidence_slider = ctk.CTkSlider(
            self.control_frame,
            from_=0.1,
            to=0.9,
            number_of_steps=8,
            command=self.update_confidence
        )
        self.confidence_slider.set(CONFIDENCE_THRESHOLD)
        self.confidence_slider.pack(pady=5)
        
        self.confidence_label = ctk.CTkLabel(
            self.control_frame,
            text=f"{CONFIDENCE_THRESHOLD*100:.0f}%"
        )
        self.confidence_label.pack()
        
        # Detection controls
        self.start_button = ctk.CTkButton(
            self.control_frame,
            text="Start Detection",
            command=self.start_detection,
            fg_color="green",
            height=40
        )
        self.start_button.pack(pady=10)
        
        self.stop_button = ctk.CTkButton(
            self.control_frame,
            text="Stop Detection",
            command=self.stop_detection,
            fg_color="red",
            height=40,
            state="disabled"
        )
        self.stop_button.pack(pady=5)
        
        # Log display
        log_frame = ctk.CTkFrame(self.control_frame)
        log_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(log_frame, text="Detection Log").pack()
        self.log_text = ctk.CTkTextbox(log_frame, height=150)
        self.log_text.pack(fill="both", expand=True)
        self.log("Application initialized")
        
        # Footer
        self.footer_frame = ctk.CTkFrame(self.root)
        self.footer_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(
            self.footer_frame,
            text="WildEye Animal Detector v1.0 | © 2023",
            text_color="gray"
        ).pack(side="right", padx=20)
        
        # Bind ESC key to quit
        self.root.bind("<Escape>", lambda e: self.root.quit())
        
        # Make video area expand when window resizes
        self.root.bind("<Configure>", self.on_window_resize)
        
    def on_window_resize(self, event):
        """Handle window resize events to maintain proper video scaling"""
        if self.current_frame:
            self.update_display(self.current_frame)
        
    def log(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        self.log_text.configure(state="normal")
        self.log_text.insert("0.0", timestamp + message + "\n")
        self.log_text.configure(state="disabled")
        
    def update_confidence(self, value):
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = value
        self.confidence_label.configure(text=f"{value*100:.0f}%")
        
    def start_detection(self):
        self.detection_active = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text="■ RUNNING", text_color="lightgreen")
        self.log("Starting animal detection...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        time.sleep(1)  # Allow camera to initialize
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.status_label.configure(text="■ ERROR", text_color="red")
            return
            
        # Load model if not loaded
        if self.model is None:
            self.log("Loading YOLOv5 model...")
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.log("Model loaded successfully")
            except Exception as e:
                self.log(f"Error loading model: {str(e)}")
                self.stop_detection()
                return
                
        # Start detection thread
        threading.Thread(target=self.detect_animals, daemon=True).start()
        
    def detect_animals(self):
        self.frame_count = 0
        self.start_time = time.time()
        
        while self.detection_active:
            ret, frame = self.cap.read()
            if not ret:
                self.log("Failed to capture frame")
                break
                
            self.frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            
            # Run detection
            results = self.model(frame)
            detections = results.pandas().xyxy[0]
            
            # Process detections
            animal_detected = False
            for _, row in detections.iterrows():
                if row['name'] in ANIMAL_CLASSES and row['confidence'] > CONFIDENCE_THRESHOLD:
                    animal_detected = True
                    self.draw_detection(frame, row)
                    if time.time() - self.last_detection_time > 1:  # Throttle logging
                        self.log(f"Detected {row['name']} ({row['confidence']:.2f})")
                        self.last_detection_time = time.time()
            
            if animal_detected and SOUND_ALERT:
                threading.Thread(target=self.play_alert_sound).start()
                
            # Calculate FPS
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.fps_label.configure(text=f"FPS: {fps:.2f}")
            
            # Update display
            self.update_display(frame)
            
        self.cleanup()
        
    def draw_detection(self, frame, detection):
        color = (0, 255, 0)  # Green
        thickness = 2
        x1, y1 = int(detection['xmin']), int(detection['ymin'])
        x2, y2 = int(detection['xmax']), int(detection['ymax'])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{detection['name']} {detection['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness)
        
    def update_display(self, frame):
        # Store current frame for resize events
        self.current_frame = frame
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Maintain aspect ratio
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if canvas_ratio > img_ratio:
            new_width = int(img_ratio * canvas_height)
            new_height = canvas_height
        else:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
            
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        photo_img = ImageTk.PhotoImage(image=resized_img)
        
        # Update canvas
        self.canvas.delete("all")
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        self.canvas.create_image(x_pos, y_pos, image=photo_img, anchor="nw")
        
        # Keep reference to prevent garbage collection
        self.photo_img = photo_img
        
    def play_alert_sound(self):
        try:
            import winsound
            winsound.Beep(1000, 200)
        except ImportError:
            print('\a', end='')  # Fallback system beep
            
    def stop_detection(self):
        self.detection_active = False
        self.status_label.configure(text="■ STOPPED", text_color="orange")
        self.log("Detection stopped by user")
        
    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    try:
        app = AnimalDetectionApp()
        app.run()
    except Exception as e:
        messagebox.showerror("Critical Error", f"The application encountered an error:\n{str(e)}")
