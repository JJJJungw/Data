from flask import Flask, request, render_template, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
import os
import tempfile
import uuid
import joblib
import tensorflow as tf

app = Flask(__name__)

# YOLO 모델 로드
model_path = "C:/Users/a/Desktop/flask/best.pt"
model = YOLO(model_path)

# LSTM 모델 및 스케일러 로드
lstm_model = tf.keras.models.load_model('lstm_model.h5')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# 현재 스크립트 위치를 기준으로 업로드 폴더 설정
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/assets/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 클래스별 색상 지정
CLASS_COLORS = {
    0: (0, 255, 0),    # 초록
    1: (255, 255, 0),  # 노랑
    2: (255, 0, 0),    # 빨강
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yolo_index')
def yolo_index():
    return render_template('YOL_index.html')

@app.route('/lstm_index')
def lstm_index():
    return render_template('lstm_index.html')

# YOLO 예측 라우트
@app.route('/predict_yolo', methods=['POST'])
def predict_yolo():
    file = request.files.get('file')
    if not file:
        return "파일이 없습니다.", 400

    ext = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        img = Image.open(file_path)
        results = model(img)
        annotated_img = np.array(img)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                color = CLASS_COLORS.get(class_id, (255, 255, 255))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        # 결과 저장
        result_image_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
        cv2.imwrite(result_image_path, annotated_img)

        return render_template('YOLO_result.html', result_image="result.jpg")

    return "유효한 이미지 파일이 아닙니다.", 400

# LSTM 예측 라우트
@app.route('/predict', methods=['POST'])
def predict_lstm():
    input_data = []
    for i in range(12):  # 12주치 입력을 받음
        week_data = [
            float(request.form[f'inTp_mean_{i+1}']),
            float(request.form[f'inTp_min_{i+1}']),
            float(request.form[f'inTp_max_{i+1}']),
            float(request.form[f'inHd_mean_{i+1}']),
            float(request.form[f'inHd_min_{i+1}']),
            float(request.form[f'inHd_max_{i+1}']),
            float(request.form[f'inCo2_mean_{i+1}']),
            float(request.form[f'inCo2_min_{i+1}']),
            float(request.form[f'inCo2_max_{i+1}']),
            float(request.form[f'outTp_mean_{i+1}']),
            float(request.form[f'outTp_min_{i+1}']),
            float(request.form[f'outTp_max_{i+1}']),
        ]
        input_data.append(week_data)

    input_data = np.array(input_data)
    input_data_scaled = scaler_X.transform(input_data)
    input_data_scaled = np.expand_dims(input_data_scaled, axis=0)

    prediction = lstm_model.predict(input_data_scaled)
    predicted_production_10평 = scaler_y.inverse_transform(prediction)[0][0]
    predicted_production_1평 = predicted_production_10평 / 10

    return render_template(
        'lstm_result.html',
        prediction_10평=predicted_production_10평,
        prediction_1평=predicted_production_1평
    )

if __name__ == '__main__':
    app.run(debug=True)
