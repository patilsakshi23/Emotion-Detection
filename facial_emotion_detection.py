import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt


checkpoint_path = 'checkpoint/best_model.keras'
final_model = tf.keras.models.load_model(checkpoint_path)

# Label to text dictionary
label_to_text = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

# Start capturing video
cap = cv2.VideoCapture(0)


window_closed = False

while not window_closed:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, -1)
        roi_gray = np.expand_dims(roi_gray, 0)
        roi_gray = roi_gray / 255.0
        
        # Predict the emotion
        predicted_class = final_model.predict(roi_gray).argmax()
        predicted_label = label_to_text[predicted_class]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Real-time Emotion Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if cv2.getWindowProperty('Real-time Emotion Detection', cv2.WND_PROP_VISIBLE) < 1:
        window_closed = True

cap.release()
cv2.destroyAllWindows()
