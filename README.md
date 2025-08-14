# KneeInsight AI 🦵🤖
An AI-powered knee health detection system that combines Deep Learning (Custom CNN, VGG16, InceptionV3) and Machine Learning (Random Forest, SVM, XGBoost) to classify knee X-ray images (Normal / Osteopenia / Osteoporosis). Includes a Flask web app for uploads and smart model selection based on per-model confidence and historical accuracy.

## Demo
![Demo Preview](asset/demo.gif)


## 🚀 Features
- **Hybrid AI stack**: 3 Deep Learning models + 3 Machine Learning models.
- **Web interface**: Flask + HTML/CSS/JS for instant predictions.
- **Smart model selection**: Chooses the most trustworthy model per image.
- **Pre-trained Models Available**: Download from the [Releases](../../releases) section.

## 📂 Project Structure
- **app.py**   # Flask web application
- **knee_detection_gg.py**     # Model training & evaluation
- **templates**     # HTML templates for the web app


## 📥 Pre-trained Models
The trained models are **not stored directly in this repository** due to size limits.  
You can download them from the [GitHub Releases](../../releases) page and place them in a `saved_models/` folder in the root directory.

Example:
- /models
- custom_cnn.h5
- vgg16_model.h5
- inceptionv3_model.h5
- random_forest.pkl
- svm_model.pkl
- xgboost_model.pkl


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
Open your browser and go to **[http://localhost:5000]**

## Upload a knee X-ray image.

Get instant classification results.


## 🔑 Keywords

Knee X-ray AI, Medical Imaging AI, CNN Knee Detection, VGG16, InceptionV3, XGBoost, Random Forest, SVM, Flask AI App, Deep Learning Healthcare, Machine Learning Orthopedics, Hybrid AI Model, Osteoporosis Detection, Orthopedic AI.


