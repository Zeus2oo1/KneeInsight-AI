from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import joblib

app = Flask(__name__)

# Global variables
models = {}
model_names = {
    'custom_cnn': 'Custom CNN',
    'vgg16': 'VGG16', 
    'inception': 'Inception',
    'random_forest': 'Random Forest',
    'svm': 'SVM',
    'xgboost': 'XGBoost'
}

model_accuracies = {
    'custom_cnn': 0.85,
    'vgg16': 0.88,
    'inception': 0.87,
    'random_forest': 0.82,
    'svm': 0.80,
    'xgboost': 0.79
}

def load_models():
    """Load all trained models"""
    try:
        model_dir = 'saved_models'
        
        # Load deep learning models - use the 'best' versions
        if os.path.exists(os.path.join(model_dir, 'custom_cnn_best.h5')):
            models['custom_cnn'] = keras.models.load_model(os.path.join(model_dir, 'custom_cnn_best.h5'))
            print(f"✅ Loaded Custom CNN (best weights)")
        elif os.path.exists(os.path.join(model_dir, 'custom_cnn_model.h5')):
            models['custom_cnn'] = keras.models.load_model(os.path.join(model_dir, 'custom_cnn_model.h5'))
            print(f"✅ Loaded Custom CNN (regular)")
        
        if os.path.exists(os.path.join(model_dir, 'vgg16_best.h5')):
            models['vgg16'] = keras.models.load_model(os.path.join(model_dir, 'vgg16_best.h5'))
            print(f"✅ Loaded VGG16 (best weights)")
        elif os.path.exists(os.path.join(model_dir, 'vgg16_model.h5')):
            models['vgg16'] = keras.models.load_model(os.path.join(model_dir, 'vgg16_model.h5'))
            print(f"✅ Loaded VGG16 (regular)")
        
        if os.path.exists(os.path.join(model_dir, 'inception_best.h5')):
            models['inception'] = keras.models.load_model(os.path.join(model_dir, 'inception_best.h5'))
            print(f"✅ Loaded Inception (best weights)")
        elif os.path.exists(os.path.join(model_dir, 'inception_model.h5')):
            models['inception'] = keras.models.load_model(os.path.join(model_dir, 'inception_model.h5'))
            print(f"✅ Loaded Inception (regular)")
        
        # Load ML models
        if os.path.exists(os.path.join(model_dir, 'random_forest_model.pkl')):
            models['random_forest'] = joblib.load(os.path.join(model_dir, 'random_forest_model.pkl'))
            print(f"✅ Loaded Random Forest")
        
        if os.path.exists(os.path.join(model_dir, 'svm_model.pkl')):
            models['svm'] = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
            print(f"✅ Loaded SVM")
        
        if os.path.exists(os.path.join(model_dir, 'xgboost_model.pkl')):
            models['xgboost'] = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))
            print(f"✅ Loaded XGBoost")
        
        print(f"✅ Loaded {len(models)} models successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def get_prediction(model, model_name, img_array):
    """Get prediction from a specific model"""
    try:
        if model_name in ['custom_cnn', 'vgg16', 'inception']:
            # Deep learning models - use full image
            print(f"🔍 Running {model_name} prediction...")
            prediction = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            print(f"✅ {model_name}: Class {predicted_class}, Confidence {confidence:.3f}")
        else:
            # ML models - need to resize to match training data
            print(f"🔍 Running {model_name} prediction...")
            
            # Resize image to 22x22 and convert to grayscale for ML models
            # 22x22 = 484 features, pad to 512
            img_small = Image.fromarray((img_array[0] * 255).astype(np.uint8))
            img_small = img_small.resize((22, 22))
            img_small = img_small.convert('L')  # Convert to grayscale
            
            # Flatten to 1D array (22*22 = 484 features)
            flattened_img = np.array(img_small).flatten() / 255.0
            
            # Pad to 512 features
            padded_img = np.zeros(512)
            padded_img[:484] = flattened_img
            flattened_img = padded_img
            
            flattened_img = np.expand_dims(flattened_img, axis=0)
            
            prediction = model.predict(flattened_img)
            predicted_class = int(prediction[0])
            
            # For ML models, we need to get confidence from prediction probabilities
            # If the model supports predict_proba, use it; otherwise use default
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(flattened_img)
                    confidence = float(np.max(proba[0]))
                else:
                    confidence = 0.85  # Default confidence for ML models
            except:
                confidence = 0.85
            
            print(f"✅ {model_name}: Class {predicted_class}, Confidence {confidence:.3f}")
        
        class_labels = ['Normal', 'Osteopenia', 'Osteoporosis']
        predicted_label = class_labels[predicted_class]
        is_normal = predicted_label == 'Normal'
        binary_result = "Normal" if is_normal else "Not Normal"
        
        return {
            'predicted_class': predicted_label,
            'binary_result': binary_result,
            'confidence': confidence,
            'is_normal': is_normal
        }
        
    except Exception as e:
        print(f"❌ Error getting prediction from {model_name}: {e}")
        return None

def calculate_smart_score(model_name, confidence, predicted_class, true_class_estimate):
    """
    Smart scoring system that considers:
    1. Confidence when prediction is likely correct
    2. Penalty when prediction is likely wrong
    3. Historical accuracy as safety net
    """
    base_accuracy = model_accuracies[model_name]
    
    # Estimate if prediction is likely correct based on confidence vs historical accuracy
    # If confidence is much higher than historical accuracy, be suspicious
    confidence_trust = min(confidence, base_accuracy + 0.1)  # Cap confidence trust
    
    # Calculate trust score (how much we trust this prediction)
    if confidence > 0.9 and confidence > base_accuracy + 0.15:
        # Very high confidence but much higher than historical accuracy - suspicious
        trust_multiplier = 0.7
    elif confidence > 0.8 and confidence > base_accuracy + 0.1:
        # High confidence but higher than historical accuracy - somewhat suspicious
        trust_multiplier = 0.85
    else:
        # Reasonable confidence - trust it
        trust_multiplier = 1.0
    
    # Final score: (trusted confidence * 0.6) + (historical accuracy * 0.4)
    trusted_confidence = confidence * trust_multiplier
    final_score = (trusted_confidence * 0.6) + (base_accuracy * 0.4)
    
    return final_score, trust_multiplier

def select_best_model_for_image(results):
    """Select best model using smart scoring system"""
    best_model = None
    best_score = -1
    
    print(f"\n🏆 Smart Model Scoring:")
    print(f"   Format: Model → Confidence → Trust Multiplier → Final Score")
    
    for model_name, result in results.items():
        if result is None:
            continue
            
        confidence = result['confidence']
        base_accuracy = model_accuracies[model_name]
        
        # Calculate smart score
        final_score, trust_multiplier = calculate_smart_score(
            model_name, confidence, result['predicted_class'], None
        )
        
        print(f"   {model_name}: {confidence:.1%} → ×{trust_multiplier:.2f} → {final_score:.3f}")
        
        if final_score > best_score:
            best_score = final_score
            best_model = model_name
    
    print(f"🎯 Winner: {best_model} (Score: {best_score:.3f})")
    return best_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            temp_path = 'temp_upload.jpg'
            file.save(temp_path)
            
            img_array = preprocess_image(temp_path)
            if img_array is None:
                return jsonify({'error': 'Error processing image'}), 400
            
            print(f"\n🚀 Starting predictions for image: {file.filename}")
            
            results = {}
            # Get predictions from all models
            for model_name, model in models.items():
                print(f"\n🔍 Processing {model_name}...")
                prediction = get_prediction(model, model_name, img_array)
                if prediction:
                    results[model_name] = prediction
                    print(f"✅ {model_name}: {prediction['binary_result']} ({prediction['confidence']:.1%})")
                else:
                    print(f"❌ {model_name}: Failed to get prediction")
            
            if not results:
                return jsonify({'error': 'No models could make predictions'}), 400
            
            # Select best model using smart scoring
            best_model = select_best_model_for_image(results)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Determine overall assessment based on best model's prediction
            best_prediction = results[best_model]
            overall_assessment = best_prediction['binary_result']
            
            response_data = {
                'success': True,
                'results': results,
                'best_model': best_model,
                'best_model_accuracy': model_accuracies[best_model], # Use confidence, not historical accuracy
                'best_model_name': model_names[best_model],
                'overall_assessment': overall_assessment
            }
            
            print(f"\n📋 Final Results:")
            print(f"   Overall Assessment: {overall_assessment}")
            print(f"   Best Model: {best_model} ({best_prediction['confidence']:.1%} confidence for this image)")
            print(f"   Best Model Prediction: {best_prediction['predicted_class']}")
            
            return jsonify(response_data)
            
    except Exception as e:
        print(f"❌ Error in upload: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Knee Detection Flask App...")
    
    if load_models():
        print("🎯 All models loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please check the saved_models directory.")