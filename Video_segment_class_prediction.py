import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from collections import Counter

def preprocess_frames(frames):
    preprocessed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, (64, 64))  # Resize frame to match model input size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color space to RGB
        frame = preprocess_input(frame)  # Preprocess frame for DenseNet model
        preprocessed_frames.append(frame)
    return np.array(preprocessed_frames)

#Load the trained model
model = tf.keras.models.load_model('C:/Users/Ronit Das/Desktop/NSUT work/Codes/Python/Video Analytics/project.keras')  # Replace with your model file path

#Define class labels
CLASS_LABELS = ['Graffiti', 'Littering', 'NormalVideos', 'PropertyDamage', 'Stealing']

#Access the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam index

#Capture frames for 10 seconds
start_time = time.time()
frames = []
while (time.time() - start_time) < 10:
    ret, frame = cap.read()
    cv2.imshow('Webcam Feed',frame)
    if not ret:
        break
    frames.append(frame)

#Preprocess frames
preprocessed_frames = preprocess_frames(frames)

#Predict class labels for the frames
predictions = model.predict(preprocessed_frames)
predicted_labels = np.argmax(predictions, axis=1)

#Select the most frequent class
predicted_class_index = Counter(predicted_labels).most_common(1)[0][0]
predicted_class = CLASS_LABELS[predicted_class_index]

#Print predicted class for the 10-second video
print("Predicted class for the 10-second video:", predicted_class)

#Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()