import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.applications import VGG16, InceptionV3
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D
from keras.optimizers import Adam, Adamax
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import shutil

# Optional imports with error handling
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Kaggle API not needed since data is already available
KAGGLE_AVAILABLE = False

# GPU Configuration
def setup_gpu():
    """Configure GPU settings for optimal performance"""
    try:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s): {gpus}")
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for faster training
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            print("✅ GPU configured successfully!")
            return True
        else:
            print("⚠️ No GPU found. Using CPU.")
            return False
    except Exception as e:
        print(f"⚠️ GPU setup failed: {e}. Using CPU.")
        return False

# Set up GPU
GPU_AVAILABLE = setup_gpu()

# Set dataset path (update to your dataset path)
data_path = '../Knee/OS Collected Data'  # Adjust path as needed
csv_path = '../Knee/Osteoporosis.csv'
model_save_dir = 'saved_models'
os.makedirs(model_save_dir, exist_ok=True)

# Create CSV file if not already present
if not os.path.exists(csv_path):
    images = []
    labels = []
    for subfolder in os.listdir(data_path):
        subfolder_path = os.path.join(data_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for image_filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_filename)
            images.append(image_path)
            labels.append(subfolder)
    df = pd.DataFrame({'image': images, 'label': labels})
    df.to_csv(csv_path, index=False)

# Load CSV
df = pd.read_csv(csv_path)

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_mapping)

# Load and preprocess images
def load_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB').resize(target_size)  # RGB for VGG16/InceptionV3
    img = img_to_array(img) / 255.0  # Normalize
    return img

df['image_data'] = df['image'].apply(load_image)

image_paths = df['image'].values
labels = df['label'].values

# Split dataset
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
print(f"Train set size: {len(train_paths)}")
print(f"Validation set size: {len(val_paths)}")

def data_generator(image_paths, labels, batch_size=16):
    while True:
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        for i in range(0, len(image_paths), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_paths = image_paths[batch_indices]
            batch_labels = labels[batch_indices]
            images = np.array([load_image(img_path) for img_path in batch_paths])
            yield images, tf.keras.utils.to_categorical(batch_labels, num_classes=3)

# Define Custom CNN
def create_custom_cnn(dropout_rate=0.0):
    input_layer = Input(shape=(224, 224, 3))
    x = Conv2D(128, (8, 8), activation='relu', padding='same')(input_layer)
    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

# Create the custom CNN model
model = create_custom_cnn()

# Print the model summary
model.summary()

# Train Custom CNN (Best Configuration: Dropout=0, Batch Size=16)
custom_cnn = create_custom_cnn(dropout_rate=0.0)
custom_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
train_gen = data_generator(train_paths, train_labels, batch_size=16)
val_gen = data_generator(val_paths, val_labels, batch_size=16)
# Create callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_dir, 'custom_cnn_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history_custom = custom_cnn.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_paths) // 16,
    validation_steps=len(val_paths) // 16,
    epochs=15,
    callbacks=callbacks
)
custom_cnn.save(os.path.join(model_save_dir, 'custom_cnn_model.h5'))

val_images = np.array([load_image(p) for p in val_paths])
custom_preds = np.argmax(custom_cnn.predict(val_images), axis=1)
print("Custom CNN Classification Report: ")
print(classification_report(val_labels, custom_preds, target_names=label_mapping.keys()))

base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_vgg16.trainable = False
x = base_model_vgg16.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)
base_model_vgg16.summary()
vgg16_model = Model(inputs=base_model_vgg16.input, outputs=output)
vgg16_model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# Create callbacks for VGG16
callbacks_vgg16 = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_dir, 'vgg16_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history_vgg16 = vgg16_model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_paths) // 16,
    validation_steps=len(val_paths) // 16,
    epochs=15,
    callbacks=callbacks_vgg16
)
vgg16_model.save(os.path.join(model_save_dir, 'vgg16_model.h5'))

# Evaluate VGG16
vgg16_preds = np.argmax(vgg16_model.predict(val_images), axis=1)
print("VGG16 Classification Report:")
print(classification_report(val_labels, vgg16_preds, target_names=label_mapping.keys()))

# Train InceptionV3
base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_inception.trainable = False
x = base_model_inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)
inception_model = Model(inputs=base_model_inception.input, outputs=output)
inception_model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# Create callbacks for InceptionV3
callbacks_inception = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_dir, 'inception_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history_inception = inception_model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=len(train_paths) // 16,
    validation_steps=len(val_paths) // 16,
    epochs=15,
    callbacks=callbacks_inception
)
inception_model.save(os.path.join(model_save_dir, 'inception_model.h5'))

# Evaluate InceptionV3
inception_preds = np.argmax(inception_model.predict(val_images), axis=1)
print("InceptionV3 Classification Report:")
print(classification_report(val_labels, inception_preds, target_names=label_mapping.keys()))

# Ensemble Model (Weighted Averaging)
def ensemble_predict_weighted(models, val_images):
    pred_custom = models[0].predict(val_images)
    pred_vgg16 = models[1].predict(val_images)
    pred_inception = models[2].predict(val_images)
    weights = [0.4, 0.3, 0.3]  # Custom CNN: 0.4, VGG16: 0.3, InceptionV3: 0.3
    ensemble_preds = np.average([pred_custom, pred_vgg16, pred_inception], axis=0, weights=weights)
    return np.argmax(ensemble_preds, axis=1)

# Ensemble Model (Majority Voting)
def ensemble_predict_majority(models, val_images):
    pred_custom = np.argmax(models[0].predict(val_images), axis=1)
    pred_vgg16 = np.argmax(models[1].predict(val_images), axis=1)
    pred_inception = np.argmax(models[2].predict(val_images), axis=1)
    stacked_preds = np.vstack([pred_custom, pred_vgg16, pred_inception])
    ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_preds)
    return ensemble_preds

# Evaluate Ensemble (Weighted Averaging)
models = [custom_cnn, vgg16_model, inception_model]
ensemble_preds_weighted = ensemble_predict_weighted(models, val_images)
print("Ensemble (Weighted Averaging) Classification Report:")
print(classification_report(val_labels, ensemble_preds_weighted, target_names=label_mapping.keys()))

# Evaluate Ensemble (Majority Voting)
ensemble_preds_majority = ensemble_predict_majority(models, val_images)
print("Ensemble (Majority Voting) Classification Report:")
print(classification_report(val_labels, ensemble_preds_majority, target_names=label_mapping.keys()))

# Confusion Matrix for Weighted Ensemble
cm = confusion_matrix(val_labels, ensemble_preds_weighted)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Ensemble (Weighted Averaging)')
plt.show()

# Plot Training History
def plot_training_history(history, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} - Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{title} - Accuracy')
    plt.show()

plot_training_history(history_custom, 'Custom CNN')
plot_training_history(history_vgg16, 'VGG16')
plot_training_history(history_inception, 'InceptionV3')

# Compare Model Accuracies
def plot_model_comparison():
    accuracies = {
        'Custom CNN': history_custom.history['val_accuracy'][-1],
        'VGG16': history_vgg16.history['val_accuracy'][-1],
        'InceptionV3': history_inception.history['val_accuracy'][-1]
    }
    plt.figure(figsize=(8, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.show()

plot_model_comparison()

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
import numpy as np

# Load CSV
df = pd.read_csv(csv_path)

# Split dataset
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

# Load images for feature extraction
train_images = np.array([load_image(p) for p in train_paths])
val_images = np.array([load_image(p) for p in val_paths])

# Feature Extraction using VGG16
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model_vgg16.input, outputs=GlobalAveragePooling2D()(base_model_vgg16.output))
train_features = feature_extractor.predict(train_images)
val_features = feature_extractor.predict(val_images)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_features, train_labels)
rf_preds = rf_model.predict(val_features)
print("Random Forest Classification Report:")
print(classification_report(val_labels, rf_preds, target_names=label_mapping.keys()))

# Save Random Forest model
import joblib
joblib.dump(rf_model, os.path.join(model_save_dir, 'random_forest_model.pkl'))
print(f"Random Forest model saved to: {os.path.join(model_save_dir, 'random_forest_model.pkl')}")

# Train SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(train_features, train_labels)
svm_preds = svm_model.predict(val_features)
print("SVM Classification Report:")
print(classification_report(val_labels, svm_preds, target_names=label_mapping.keys()))

# Save SVM model
joblib.dump(svm_model, os.path.join(model_save_dir, 'svm_model.pkl'))
print(f"SVM model saved to: {os.path.join(model_save_dir, 'svm_model.pkl')}")

# Train XGBoost
if XGBOOST_AVAILABLE:
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(train_features, train_labels)
    xgb_preds = xgb_model.predict(val_features)
    print("XGBoost Classification Report:")
    print(classification_report(val_labels, xgb_preds, target_names=label_mapping.keys()))
    
    # Save XGBoost model
    joblib.dump(xgb_model, os.path.join(model_save_dir, 'xgboost_model.pkl'))
    print(f"XGBoost model saved to: {os.path.join(model_save_dir, 'xgboost_model.pkl')}")
else:
    print("XGBoost is not available, skipping.")

# Confusion Matrix for XGBoost (as an example)
if XGBOOST_AVAILABLE:
    cm = confusion_matrix(val_labels, xgb_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - XGBoost')
    plt.show()

# Compare ML Model Accuracies
def plot_ml_model_comparison():
    accuracies = {
        'Random Forest': accuracy_score(val_labels, rf_preds),
        'SVM': accuracy_score(val_labels, svm_preds),
    }
    if XGBOOST_AVAILABLE:
        accuracies['XGBoost'] = accuracy_score(val_labels, xgb_preds)
    plt.figure(figsize=(8, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.title('ML Model Accuracy Comparison')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.tight_layout()
    plt.show()

plot_ml_model_comparison()

print("ML Model Validation Accuracies:")
print(f"Random Forest: {accuracy_score(val_labels, rf_preds):.4f}")
print(f"SVM: {accuracy_score(val_labels, svm_preds):.4f}")
if XGBOOST_AVAILABLE:
    print(f"XGBoost: {accuracy_score(val_labels, xgb_preds):.4f}")

"""**Reasoning**:
Calculate the false positive rate (fpr), true positive rate (tpr), and area under the ROC curve (AUC) for each trained model and each class using the validation data.


"""

from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC


models = {
    'Custom CNN': custom_cnn,
    'VGG16': vgg16_model,
    'InceptionV3': inception_model,
    'Random Forest': rf_model,
    'SVM': svm_model,
}
if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb_model

fpr_dict = {}
tpr_dict = {}
auc_dict = {}

class_names = list(label_mapping.keys())

for model_name, model in models.items():
    fpr_dict[model_name] = {}
    tpr_dict[model_name] = {}
    auc_dict[model_name] = {}

    if model_name in ['Random Forest', 'SVM', 'XGBoost']:
        y_prob = model.predict_proba(val_features)
    else:
        y_prob = model.predict(val_images)


    for i in range(len(class_names)):
        y_true_binary = (val_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        fpr_dict[model_name][class_names[i]] = fpr
        tpr_dict[model_name][class_names[i]] = tpr
        auc_dict[model_name][class_names[i]] = roc_auc

print("AUC for each model and class:")
for model_name, aucs in auc_dict.items():
    print(f"\n{model_name}:")
    for class_name, class_auc in aucs.items():
        print(f"  {class_name}: {class_auc:.4f}")

def plot_individual_roc_curves_side_by_side(fpr_dict, tpr_dict, auc_dict, model_names, class_names):
    num_models = len(model_names)
    num_rows = (num_models + 1) // 3  # Calculate number of rows needed for two columns

    fig, axes = plt.subplots(num_rows, 3, figsize=(16, 7 * num_rows))
    axes = axes.flatten() # Flatten the axes array for easy iteration

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        for class_name in class_names:
            ax.plot(fpr_dict[model_name][class_name], tpr_dict[model_name][class_name],
                    label=f'{class_name} (AUC = {auc_dict[model_name][class_name]:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc='lower right')
        ax.grid(True)

    # Hide any unused subplots if the number of models is odd
    if num_models % 2 != 0:
        fig.delaxes(axes[num_models])

    plt.tight_layout()
    plt.show()

# Plot individual ROC curves for each model side by side
plot_individual_roc_curves_side_by_side(fpr_dict, tpr_dict, auc_dict, list(models.keys()), class_names)



"""**Reasoning**:
Calculate and display the number of images in each class to understand the current distribution.


"""

class_distribution = df['label'].value_counts()
print("Class Distribution:")
print(class_distribution)

"""**Reasoning**:
Create a bar plot to visualize the class distribution using the `class_distribution` Series.


"""

plt.figure(figsize=(8, 6))
sns.barplot(x=class_distribution.index, y=class_distribution.values)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Distribution of Images per Class')
plt.xticks(ticks=list(label_mapping.values()), labels=list(label_mapping.keys()))
plt.show()

"""**Reasoning**:
Explain what data imbalance is and why it's a problem in machine learning, particularly for image datasets, and describe data augmentation as a technique to balance image datasets, including common augmentation techniques relevant to images.


"""

print("Data imbalance occurs when the number of samples in different classes of a dataset is significantly unequal. This is a common problem in machine learning, especially in classification tasks. In the context of image datasets, if one class has many more images than others, a model trained on this data may become biased towards the majority class, leading to poor performance on minority classes.")
print("\nData augmentation is a powerful technique to address class imbalance in image datasets. It involves creating new, synthetic training examples by applying various transformations to the existing images in the dataset. This helps to increase the number of images in minority classes, making the dataset more balanced without collecting new data.")
print("\nCommon data augmentation techniques for images include:")
print("- Rotation: Rotating the image by a certain angle.")
print("- Flipping: Horizontally or vertically flipping the image.")
print("- Zooming: Randomly zooming in or out of the image.")
print("- Shifting: Shifting the image horizontally or vertically.")
print("- Shearing: Applying a shearing transformation.")
print("- Brightness and Contrast Adjustment: Altering the brightness and contrast of the image.")
print("- Adding Noise: Introducing random noise to the image.")

"""**Reasoning**:
Describe oversampling and undersampling techniques for handling class imbalance, their applicability to image datasets, and their potential drawbacks.


"""

print("\nOversampling techniques aim to increase the number of samples in the minority class by creating duplicate or synthetic samples. While techniques like SMOTE (Synthetic Minority Over-sampling Technique) are popular for tabular data by generating synthetic samples in the feature space, they are not directly applicable to image data in its pixel form. However, image-specific oversampling can be achieved through advanced data augmentation or generating new images using techniques like Generative Adversarial Networks (GANs).")
print("\nUndersampling techniques aim to reduce the number of samples in the majority class to match the number of samples in the minority class. This can be done randomly or based on certain criteria (e.g., removing samples that are close to the decision boundary). While simple to implement, a major drawback of undersampling is the potential loss of important information contained in the discarded majority class samples, which can negatively impact the model's performance.")

"""**Reasoning**:
Briefly discuss other potential strategies and considerations for handling class imbalance in image datasets, such as using different evaluation metrics or adjusting class weights during model training, and finish the task.


"""

print("\nBesides data-level techniques, other strategies for handling class imbalance in image datasets include:")
print("- Using appropriate evaluation metrics: Instead of relying solely on accuracy, which can be misleading in imbalanced datasets, consider metrics like F1-score, precision, recall, and AUC, which provide a more nuanced view of model performance on each class.")
print("- Adjusting class weights: During model training, assign higher weights to the minority classes and lower weights to the majority classes. This tells the model to pay more attention to correctly classifying samples from the minority classes.")
print("- Using different loss functions: Some loss functions are more robust to class imbalance than standard cross-entropy.")
print("- Ensemble methods: Combining multiple models can sometimes improve performance on imbalanced datasets.")

"""**Reasoning**:
Based on the class distribution calculated previously, the dataset is imbalanced. The 'Osteoporosis' class has significantly fewer images than 'Normal' and 'Osteopenia'. Data augmentation is a suitable technique for image datasets to balance the classes by generating synthetic images for the minority classes. I will implement data augmentation for the minority class ('Osteopenia').


"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Determine if balancing is needed
# We have 780 Normal, 374 Osteopenia, and 793 Osteoporosis images
# Balancing is needed as 'Osteopenia' has significantly fewer images

# Choose an appropriate technique: Data Augmentation for the minority class ('Osteopenia')
# The minority class is 'Osteopenia' (label 1) with 374 images.
# The target number of images for the minority class can be set closer to the majority classes (e.g., around 780 or 793).
target_minority_count = 780

# Filter the DataFrame to get minority class images
minority_df = df[df['label'] == 1].copy()

# Calculate how many new images to generate
num_augmentations_needed = target_minority_count - len(minority_df)
print(f"Number of Osteopenia images to generate: {num_augmentations_needed}")

# Set up ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Convert minority class images to numpy array for augmentation
minority_images_array = np.array([load_image(img_path) for img_path in minority_df['image'].values])

# Generate augmented images and their labels
augmented_images = []
augmented_labels = []

# Use flow method to generate augmented images
# Need to fit the generator to the data
datagen.fit(minority_images_array)

# Use flow method to generate augmented images
# Generate batches until we have enough images
generated_count = 0
for x_batch in datagen.flow(minority_images_array, batch_size=1, shuffle=False):
    augmented_images.append(x_batch[0])
    augmented_labels.append(1) # Label for Osteopenia
    generated_count += 1
    if generated_count >= num_augmentations_needed:
        break

# Convert augmented images to the format used in the original DataFrame
augmented_image_data = [img for img in augmented_images]
augmented_image_paths = [f'/generated_image_{i}.png' for i in range(num_augmentations_needed)] # Placeholder paths

# Create a DataFrame for augmented data
augmented_df = pd.DataFrame({'image': augmented_image_paths, 'label': augmented_labels, 'image_data': augmented_image_data})

# Combine original and augmented data
balanced_df = pd.concat([df, augmented_df], ignore_index=True)

# Verify the result: Check the new class distribution
print("\nNew Class Distribution after Augmentation:")
print(balanced_df['label'].value_counts())

# Update the global df to the balanced_df for subsequent steps
df = balanced_df

"""**Reasoning**:
Calculate and display the updated class distribution and then visualize it using a bar plot.


"""

updated_class_distribution = df['label'].value_counts()
print("Updated Class Distribution:")
print(updated_class_distribution)

plt.figure(figsize=(8, 6))
sns.barplot(x=updated_class_distribution.index, y=updated_class_distribution.values)
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Updated Distribution of Images per Class')
plt.xticks(ticks=list(label_mapping.values()), labels=list(label_mapping.keys()))
plt.show()

"""**Reasoning**:
Train the Custom CNN model using the balanced training data.


"""

# Define Custom CNN
def create_custom_cnn(dropout_rate=0.0):
    input_layer = Input(shape=(224, 224, 3))
    x = Conv2D(128, (8, 8), activation='relu', padding='same')(input_layer)
    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

# Update data_generator to use 'image_data' column directly and handle list of arrays
def data_generator_updated(dataframe, batch_size=16):
    while True:
        indices = np.arange(len(dataframe))
        np.random.shuffle(indices)
        for i in range(0, len(dataframe), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_df = dataframe.iloc[batch_indices]
            batch_images_data = batch_df['image_data'].values
            batch_labels = batch_df['label'].values

            # Convert list of arrays to a single numpy array
            # Ensure consistent shapes if necessary, though load_image should handle this
            images = np.array(list(batch_images_data))


            yield images, tf.keras.utils.to_categorical(batch_labels, num_classes=3)

# Load and preprocess images
def load_image(image_path, target_size=(224, 224)):
    # Check if the path is a placeholder for augmented data
    # This case should not happen with the corrected validation data loading
    # but kept for clarity if any issues arise
    if image_path.startswith('/generated_image_'):
        return None # Or handle appropriately

    try:
        img = Image.open(image_path).convert('RGB').resize(target_size)  # RGB for VGG16/InceptionV3
        img = img_to_array(img) / 255.0  # Normalize
        return img
    except Exception as e:
        print(f"Error loading image: {image_path}, Error: {e}")
        return None # Return None for problematic images


# Train Custom CNN (Best Configuration: Dropout=0, Batch Size=16)
custom_cnn = create_custom_cnn(dropout_rate=0.0)
custom_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators using the balanced DataFrame for training and original validation data
train_gen_balanced = data_generator_updated(df, batch_size=16)

# For validation, create a DataFrame from original val_paths and val_labels
# Load image data for validation set
val_images_data = [load_image(p) for p in val_paths]
# Filter out None values in case of loading errors
val_paths_filtered = [val_paths[i] for i, img_data in enumerate(val_images_data) if img_data is not None]
val_labels_filtered = [val_labels[i] for i, img_data in enumerate(val_images_data) if img_data is not None]
val_images_data_filtered = [img_data for img_data in val_images_data if img_data is not None]


val_df_original = pd.DataFrame({'image_data': val_images_data_filtered, 'label': val_labels_filtered})


val_gen_original = data_generator_updated(val_df_original, batch_size=16)

print(f"Number of validation images after filtering: {len(val_df_original)}")

history_custom_balanced = custom_cnn.fit(
    train_gen_balanced,
    validation_data=val_gen_original,
    steps_per_epoch=len(df) // 16,
    validation_steps=len(val_df_original) // 16, # Use filtered validation size
    epochs=15,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
custom_cnn.save(os.path.join(model_save_dir, 'custom_cnn_balanced_model.h5'))

# Train VGG16 model
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_vgg16.trainable = False
x = base_model_vgg16.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)
vgg16_model_balanced = Model(inputs=base_model_vgg16.input, outputs=output)
vgg16_model_balanced.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history_vgg16_balanced = vgg16_model_balanced.fit(
    train_gen_balanced,
    validation_data=val_gen_original,
    steps_per_epoch=len(df) // 16,
    validation_steps=len(val_df_original) // 16, # Use filtered validation size
    epochs=15,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
vgg16_model_balanced.save(os.path.join(model_save_dir, 'vgg16_balanced_model.h5'))

# Train InceptionV3 model
base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_inception.trainable = False
x = base_model_inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)
inception_model_balanced = Model(inputs=base_model_inception.input, outputs=output)
inception_model_balanced.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

history_inception_balanced = inception_model_balanced.fit(
    train_gen_balanced,
    validation_data=val_gen_original,
    steps_per_epoch=len(df) // 16,
    validation_steps=len(val_df_original) // 16, # Use filtered validation size
    epochs=15,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)
inception_model_balanced.save(os.path.join(model_save_dir, 'inception_balanced_model.h5'))

