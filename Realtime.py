import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

#Function to preprocess the webcam frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize frame to match model input size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color space to RGB
    frame = preprocess_input(frame)  # Preprocess frame for DenseNet model
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

#Function to predict class label and draw on frame
def predict_and_draw(frame, model, labels):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    predicted_class_index = np.argmax(predictions)
    predicted_class = labels[predicted_class_index]
    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

#Load the trained model
model = tf.keras.models.load_model('C:\\Users\\Ronit Das\\Desktop\\NSUT work\\Codes\\Python\\Video Analytics\\project.keras')  # Replace with your model file path

#Define class labels
CLASS_LABELS = ['Graffiti', 'Littering', 'NormalVideos', 'PropertyDamage', 'Stealing']

#Access the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict class label and draw on frame
    frame_with_prediction = predict_and_draw(frame, model, CLASS_LABELS)

    # Display the frame
    cv2.imshow('Webcam Feed', frame_with_prediction)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
