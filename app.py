from flask import Flask, request, render_template, jsonify
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# สร้างโฟลเดอร์สำหรับเก็บรูปที่ upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable สำหรับเก็บ model
model = None

def load_model_lazy():
    """โหลด model เมื่อต้องใช้งานจริง (lazy loading)"""
    global model
    if model is None:
        print("กำลังโหลด model จาก my_model_new.h5...")
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # แก้ปัญหา batch_shape และ DTypePolicy compatibility
            original_from_config = keras.layers.InputLayer.from_config
            
            @classmethod
            def patched_from_config(cls, config):
                # แปลง batch_shape -> batch_input_shape
                if 'batch_shape' in config:
                    config['batch_input_shape'] = config.pop('batch_shape')
                return original_from_config.__func__(cls, config)
            
            # ใช้ patched version
            keras.layers.InputLayer.from_config = patched_from_config
            
            try:
                # โหลดด้วย custom_object_scope เพื่อแก้ปัญหา DTypePolicy
                with keras.utils.custom_object_scope({'DTypePolicy': keras.mixed_precision.Policy}):
                    model = keras.models.load_model('ChestXRayModel.h5', compile=False)
                print("✅ โหลด model สำเร็จ!")
                print(f"📊 Model input shape: {model.input_shape}")
            finally:
                # คืนค่าเดิม
                keras.layers.InputLayer.from_config = original_from_config
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")
            return None
    return model

# กำหนดประเภทไฟล์ที่รองรับ
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img):
    """เตรียมรูปภาพสำหรับการ predict - ตรงกับที่ใช้ใน Colab"""
    # แปลงเป็น grayscale ถ้ายังไม่เป็น
    if img.mode != 'L':
        img = img.convert('L')
    
    # ปรับขนาดรูปภาพเป็น 224x224 (ตาม input shape ของ model)
    img = img.resize((224, 224))
    
    # แปลงเป็น array
    img_array = np.array(img)
    
    # Normalize ค่า pixel (0-255 -> 0-1)
    img_array = img_array / 255.0
    
    # เพิ่ม dimension ให้เป็น (1, 224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # โหลด model (lazy loading)
            current_model = load_model_lazy()
            if current_model is None:
                return jsonify({'error': 'ไม่สามารถโหลด model ได้ โปรดตรวจสอบการติดตั้ง TensorFlow'}), 500
            
            # อ่านรูปภาพ
            img = Image.open(io.BytesIO(file.read()))
            
            # เตรียมรูปภาพ (จะแปลงเป็น grayscale ใน prepare_image)
            processed_img = prepare_image(img)
            
            # ทำนาย
            prediction = current_model.predict(processed_img)
            
            # กำหนด class labels ตามที่ train มา
            # class_indices = {'COVID19': 0, 'NORMAL': 1, 'PNEUMONIA': 2, 'TURBERCULOSIS': 3}
            class_labels = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
            
            # หา class ที่มีความน่าจะเป็นสูงสุด
            pred_class = np.argmax(prediction[0])
            confidence_percent = float(prediction[0][pred_class]) * 100
            result = class_labels[pred_class]
            
            # สร้าง dictionary สำหรับแสดงความน่าจะเป็นทุก class
            all_predictions = {
                class_labels[i]: round(float(prediction[0][i]) * 100, 2) 
                for i in range(len(class_labels))
            }
            
            print(f"🔍 ผลการทำนายดิบ: {prediction[0]}")
            print(f"🔍 คลาสที่ทำนาย: {result} ({pred_class}) - Confidence: {confidence_percent:.2f}%")
            print(f"🔍 ทุก class: {all_predictions}")
            
            return jsonify({
                'prediction': result,
                'confidence': round(confidence_percent, 2),
                'all_predictions': all_predictions
            })
            
        except Exception as e:
            return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500
    
    return jsonify({'error': 'ไฟล์ไม่ถูกต้อง กรุณาอัปโหลดไฟล์ .png, .jpg หรือ .jpeg'}), 400

if __name__ == '__main__':
    print("เริ่มต้น Chest X-Ray Web Application...")
    print("เปิดเว็บได้ที่: http://localhost:5001")
    app.run(debug=False, host='0.0.0.0', port=5001)
