import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained emotion detection model
model = load_model('model.h5')

# Create the main application window
app = tk.Tk()
app.title("Emotion Detection App")

# Create a Canvas to display the camera feed
canvas = tk.Canvas(app, width=640, height=480)
canvas.pack()

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to start emotion detection
def start_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = gray[y:y + h, x:x + w]

            # Resize the face image for the model
            face_img = cv2.resize(face_roi, (48, 48))
            face_img = np.reshape(face_img, (1, 48, 48, 1)) / 255.0

            # Predict emotion
            emotion_pred = model.predict(face_img)
            emotion_label = np.argmax(emotion_pred)
            emotion = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][emotion_label]

            # Draw a green square around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the detected emotion
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame in the GUI
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img
        app.update()

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create a button to start emotion detection
start_button = tk.Button(app, text="Start Detection", command=start_detection)
start_button.pack()

# Function to handle the closing of the app
def on_close():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.destroy()

# Set the close event handler
app.protocol("WM_DELETE_WINDOW", on_close)

# Run the application
app.mainloop()
