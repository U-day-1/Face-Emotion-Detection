from flask import Flask, render_template, Response, request
from flask_cors import CORS

import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load your trained emotion detection model
model = load_model('model.h5')

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the WebSocket server
ws = None

def detect_emotion(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face image for the model
        face_img = cv2.resize(face_roi, (48, 48))
        face_img = face_img.reshape((1, 48, 48, 1))
        face_img = face_img / 255.0

        # Predict emotion
        emotion_pred = model.predict(face_img)
        emotion_label = np.argmax(emotion_pred)
        emotion = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][emotion_label]

        # Draw a green square around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the detected emotion
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Return the emotion if faces are detected
        return frame, emotion

    # Return the original frame if no faces are detected
    return frame, None

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame, detected_emotion = detect_emotion(frame)

        # If emotion is detected, send it through WebSocket
        if ws and detected_emotion:
            ws.send(detected_emotion)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ws')
def ws_route():
    global ws
    ws = request.environ.get('wsgi.websocket')
    return ''  # WebSocket connection established

if __name__ == '__main__':
    app.run(debug=True)
