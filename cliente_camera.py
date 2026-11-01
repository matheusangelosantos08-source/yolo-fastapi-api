import cv2
import requests
import threading

# URL da sua API local
URL = "http://127.0.0.1:8000/detect"

cap = cv2.VideoCapture(0)
latest_detections = []
is_sending = False

def send_frame(frame_data):
    global latest_detections, is_sending
    try:
        _, img_encoded = cv2.imencode('.jpg', frame_data)
        response = requests.post(URL, files={"file": img_encoded.tobytes()}, timeout=3)
        if response.status_code == 200:
            data = response.json()
            print("Resposta do servidor:", data)
            latest_detections = data.get("detections", [])
    except Exception as e:
        print("Erro no envio:", e)
    finally:
        is_sending = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o frame da câmara.")
        break

    if not is_sending:
        is_sending = True
        frame_copy = frame.copy()
        threading.Thread(target=send_frame, args=(frame_copy,), daemon=True).start()

    # Desenha as caixas se existirem deteções
    for det in latest_detections:
        box = det.get("box")
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Camera - Cloud YOLO (Fluido)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()