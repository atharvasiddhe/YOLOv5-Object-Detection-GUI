#Author:Atharva Siddhe 
#Object detection using yolov5 model 
import torch
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # Set the model to evaluation mode

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Recognition")
        self.root.geometry("800x600")

        self.capture = None
        self.img_label = Label(self.root)
        self.img_label.pack(pady=10)

        self.button_frame = Frame(self.root)
        self.button_frame.pack(pady=10)

        self.upload_button = Button(self.button_frame, text="Upload Image", command=self.open_filechooser)
        self.upload_button.grid(row=0, column=0, padx=5)

        self.start_button = Button(self.button_frame, text="Start Live Camera Detection", command=self.start_detection)
        self.start_button.grid(row=0, column=1, padx=5)

        self.stop_button = Button(self.button_frame, text="Stop Detection", command=self.stop_detection)
        self.stop_button.grid(row=0, column=2, padx=5)

        self.label = Label(self.root, text="Detected Objects")
        self.label.pack(pady=10)

        self.text_box = Text(self.root, height=10, width=80, state='disabled')
        self.text_box.pack(pady=10)

    def open_filechooser(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            print("Error: Could not read image.")
            return
        self.run_detection(frame)

    def start_detection(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                print("Error: Could not open webcam.")
                return
        self.update()

    def stop_detection(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def update(self):
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            print("Error: Could not read frame.")
            return

        self.run_detection(frame)
        self.root.after(33, self.update)  # Update every 33 ms (~30 FPS)

    def run_detection(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)  # Get predictions

        detected_objects = []

        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            label = model.names[int(cls)]
            if conf > 0.5:  # Confidence threshold
                detected_objects.append(f"{label} {conf:.2f}")
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Put label near bounding box
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update detected objects list in the text box
        self.text_box.config(state='normal')
        self.text_box.delete(1.0, END)
        self.text_box.insert(END, "\n".join(detected_objects))
        self.text_box.config(state='disabled')

        # Display the frame in the Tkinter window
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.img_label.imgtk = imgtk
        self.img_label.configure(image=imgtk)

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

if __name__ == '__main__':
    root = Tk()
    app = YOLOApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
