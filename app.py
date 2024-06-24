from flask import Flask, render_template, Response
import cv2
import requests
import numpy as np
import torch
from ultralytics import YOLO

app = Flask(__name__)

# Dirección IP de la ESP32-CAM y URL de la transmisión de video
ESP32_CAM_IP = '192.168.0.105'
ESP32_CAM_URL = f'http://{ESP32_CAM_IP}:80/'

# Cargar el modelo YOLOv8
model = YOLO('modelin.pt')

def generate_frames():
    while True:
        try:
            # Obtener el video de la cámara
            response = requests.get(ESP32_CAM_URL, stream=True)
            if response.status_code == 200:
                bytes_data = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        # Convertir los bytes en una imagen OpenCV
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                        # Realizar la detección de objetos
                        results = model(frame)

                        # Dibujar las detecciones en el frame
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas del cuadro delimitador
                                conf = box.conf[0]  # Confianza
                                cls = int(box.cls[0])  # Clase
                                label = f'{model.names[cls]} {conf:.2f}'
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print("Error:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
