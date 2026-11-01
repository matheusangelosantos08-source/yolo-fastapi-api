import cv2
import requests
import threading
import time

# Servidor no Render (com a barra no final, para evitar o redirect 307)
BASE_URL = "https://yolo-fastapi-api-65pe.onrender.com"
URL = BASE_URL + "/detect/"

# Tamanho da imagem enviada ao servidor (menor = mais leve e mais rápido)
SEND_WIDTH, SEND_HEIGHT = 320, 240
# Intervalo mínimo entre envios, em segundos
SEND_INTERVAL = 0.3
# Qualidade do JPEG (0-100)
JPEG_QUALITY = 60
# Tempo máximo de espera por resposta (o plano free do Render é lento)
REQUEST_TIMEOUT = 30

cap = cv2.VideoCapture(0)
latest_detections = []
is_sending = False
server_ready = False
last_sent = 0.0


def wake_server():
    """Acorda o servidor do Render (plano free dorme após inatividade)."""
    global server_ready
    print("Acordando o servidor... pode levar até ~1 minuto.")
    try:
        r = requests.get(BASE_URL + "/", timeout=120)
        if r.status_code == 200:
            print("Servidor pronto!")
            server_ready = True
        else:
            print("Servidor respondeu com status", r.status_code)
    except Exception as e:
        print("Erro ao acordar o servidor:", e)


def send_frame(frame_data, scale_x, scale_y):
    global latest_detections, is_sending
    try:
        _, img_encoded = cv2.imencode(
            '.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        response = requests.post(
            URL, files={"file": img_encoded.tobytes()}, timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            detections = data.get("detections", [])

            # Converte as caixas do tamanho reduzido para o tamanho original
            for det in detections:
                box = det.get("box")
                if box:
                    x1, y1, x2, y2 = box
                    det["box"] = [
                        x1 * scale_x, y1 * scale_y,
                        x2 * scale_x, y2 * scale_y,
                    ]
            latest_detections = detections
        else:
            print("Servidor respondeu com status", response.status_code)
    except Exception as e:
        print("Erro no envio:", e)
    finally:
        is_sending = False


threading.Thread(target=wake_server, daemon=True).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o frame da câmara.")
        break

    now = time.time()
    if server_ready and not is_sending and now - last_sent > SEND_INTERVAL:
        last_sent = now
        is_sending = True

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (SEND_WIDTH, SEND_HEIGHT))
        scale_x = w / SEND_WIDTH
        scale_y = h / SEND_HEIGHT

        threading.Thread(
            target=send_frame, args=(small, scale_x, scale_y), daemon=True
        ).start()

    # Desenha as caixas se existirem deteções
    for det in latest_detections:
        box = det.get("box")
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not server_ready:
        cv2.putText(frame, "Acordando servidor...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Camera - Cloud YOLO (Fluido)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()