from original_train import DataSetFactory  # Import the modified DataSetFactory
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Instantiate DataSetFactory and get training data
factory = DataSetFactory()
train_images, train_emotions = factory.get_training_data()

# Emotion Distribution
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_counts = np.bincount(train_emotions)
sns.barplot(x=emotion_labels, y=emotion_counts)
plt.title("Distribution of Emotions in Training Set")
plt.show()

# Show Sample Images
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(train_images[i], cmap='gray')
    ax.set_title(emotion_labels[train_emotions[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()

# Pixel Intensity Distribution
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    img_array = np.asarray(train_images[i])
    sns.histplot(img_array.flatten(), ax=ax, kde=True)
    ax.set_title(f'Image {i+1} Intensity')
plt.tight_layout()
plt.show()
