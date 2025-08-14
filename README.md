# KneeInsight AI 🦵🤖
An AI-powered knee health detection system that uses CNN and transfer learning (VGG16, InceptionV3) to classify knee X-ray images for early detection of potential issues.

## 🚀 Features
- **Deep Learning Models**: CNN, VGG16, and InceptionV3 architectures trained on knee X-ray datasets.
- **Web Interface**: Flask-based app for easy image upload and instant predictions.
- **Multiple Model Support**: Choose between CNN, VGG16, or InceptionV3 for inference.
- **Pre-trained Models Available**: Download from the [Releases](../../releases) section.

## 📂 Project Structure
- **app.py # Flask web application
- **knee_detection_gg.py # Model training & evaluation
- **templates/ # HTML templates for the web app


## 📥 Pre-trained Models
The trained models are **not stored directly in this repository** due to size limits.  
You can download them from the [GitHub Releases](../../releases) page and place them in a `models/` folder in the root directory.

Example:
- **/models
- **cnn_model.h5
- **vgg16_model.h5
- **inceptionv3_model.h5


## 🛠 Installation
1. Clone the repository:
```bash
git clone https://github.com/Zeus2oo1/KneeInsight-AI.git
cd KneeInsight-AI
```
## Install dependencies
```bash
pip install -r requirements.txt
```

Download pre-trained models from the Releases page.


Run the app:
```bash
python app.py
```


## 🖼 Usage
Open your browser and go to Localhost

## Upload a knee X-ray image.

Get instant classification results.


## 🔑 Keywords

Knee X-ray AI, Medical Imaging AI, CNN Knee Detection, VGG16, InceptionV3, Flask AI App, Deep Learning Healthcare, Knee Health Prediction, Orthopedic AI.


