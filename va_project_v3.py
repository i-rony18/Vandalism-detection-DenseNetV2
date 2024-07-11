import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory, video_to_array  

# Define video paths and class labels
train_data_dir = "C:/Users/Ronit Das/Desktop/NSUT work/Video Analytics/Vandalism detection/Video Dataset/Test data"
test_data_dir = "C:/Users/Ronit Das/Desktop/NSUT work/Video Analytics/Vandalism detection/Video Dataset/Train data"
class_labels = ["Graffiti", "Littering", "NormalVideos", "PropertyDamage", "Stealing"]

# Define function to load and preprocess videos
def load_video(path, target_size=(224, 224)):
  # Load video using video.load_from_path
  vid = video.load_from_path(path)
  # Extract frames and preprocess (resize, normalize)
  frames = video.extract_frames(vid)
  # Select a subset of frames (e.g., every 5th frame)
  frames = frames[::5]
  # Reshape frames for the model input
  frames = np.expand_dims(frames, axis=0)  # Add batch dimension
  frames = video.preprocess_input(frames)  # Normalize for InceptionV3
  return frames

# Load training and testing data
train_features = []
train_labels = []
for label in class_labels:
  class_dir = os.path.join(train_data_dir, label)
  for filename in os.listdir(class_dir):
    video_path = os.path.join(class_dir, filename)
    features = load_video(video_path)
    train_features.append(features)
    train_labels.append(class_labels.index(label))  # Convert label to index

# Similar approach for loading testing data

# Create the model using a pre-trained InceptionV3
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(frames.shape[1], frames.shape[2], frames.shape[3]))
for layer in base_model.layers:
  layer.trainable = False  # Freeze pre-trained layers

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)  # Add a dense layer for learning
x = Dropout(0.5)(x)  # Dropout for regularization
predictions = Dense(len(class_labels), activation="softmax")(x)  # Output layer with softmax for multi-class

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=3)

# Train the model
model.fit(np.array(train_features), 
          to_categorical(train_labels), 
          epochs=10, 
          validation_split=0.2, 
          callbacks=[early_stopping])

# Evaluate the model on test data
# (similar approach as training but using test data)

# Use the trained model to classify new videos
def classify_video(video_path):
  # Load and preprocess video
  features = load_video(video_path)
  # Predict class probabilities
  predictions = model.predict(features)
  # Get the predicted class label
  predicted_class = class_labels[np.argmax(predictions[0])]
  return predicted_class

# Example usage
new_video_path = "path/to/new/video.mp4"
predicted_class = classify_video(new_video_path)
print(f"Predicted class: {predicted_class}")