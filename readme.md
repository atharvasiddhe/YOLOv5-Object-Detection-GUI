
--------------------------------------------------------------------------------------------------------------------------

Object Detection Using YOLOv5

--------------------------------------------------------------------------------------------------------------------------
Author: Atharva Siddhe
--------------------------------------------------------------------------------------------------------------------------
#Table of Contents

--Introduction
--Features
--Requirements
--Installation
--Usage
--Project Structure
--Acknowledgments
--------------------------------------------------------------------------------------------------------------------------
#Introduction

This project demonstrates object detection using the YOLOv5 model in a Tkinter-based GUI application. Users can upload an image for object detection or use a live camera feed to detect objects in real-time.

--------------------------------------------------------------------------------------------------------------------------
#Features

--Upload an image and detect objects within it.
--Use a live camera feed to detect objects in real-time.
--Display detected objects along with their confidence scores.
--Show bounding boxes around detected objects.

--------------------------------------------------------------------------------------------------------------------------
#Requirements

--Python 3.x
--PyTorch
--OpenCV
--Tkinter
--PIL (Pillow)
--YOLOv5 model from Ultralytics

--------------------------------------------------------------------------------------------------------------------------
#Installation

1)Clone the repository:
git clone https://github.com/yourusername/yourproject.git
cd yourproject

2)Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3)Install the required packages:
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install tkinter

--------------------------------------------------------------------------------------------------------------------------
#usage

1)Run the application:
In terminal type
python detect.py

2)In the GUI :
--Click "Upload Image" to select and process an image file.
--Click "Start Live Camera Detection" to begin real-time object detection using your webcam.
--Click "Stop Detection" to stop the live camera feed.

--------------------------------------------------------------------------------------------------------------------------
#Project Structure

yourproject/
│
├── main.py           # The main script to run the application
├── README.md         # This readme file
└── requirements.txt  # List of required packages (optional)

--------------------------------------------------------------------------------------------------------------------------
#Acknowledgments:

--Ultralytics for the YOLOv5 model.
--OpenCV for image processing.
--Tkinter for the GUI framework.
--Pillow for image handling in Python.
--------------------------------------------------------------------------------------------------------------------------
NOTE:This project is created for educational purposes and to demonstrate the integration of deep learning models with a simple GUI application.

--------------------------------------------------------------------------------------------------------------------------
