import cv2
import requests

# URL da sua API FastAPI
url_api = "http://127.0.0.1:8000/detect/"

# Se for testar com a webcam do PC, use 0. 
# Se for câmara IP, substitua pelo link RTSP ou HTTP (ex: "rtsp://utilizador:senha@ip_da_camera:554/stream")
fonte_camera = 0 

cap = cv2.VideoCapture(fonte_camera)

if not cap.isOpened():
    print("Erro: Não foi possível aceder à câmara.")
    exit()

print("Câmara ligada! A enviar frames para a API... Pressione 'q' para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o frame da câmara.")
        break

    # Codifica o frame em formato JPEG
    sucesso, img_encoded = cv2.imencode('.jpg', frame)
    if not sucesso:
        continue

    try:
        # Envia para a API via POST
        response = requests.post(
            url_api, 
            files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        )
        
        if response.status_code == 200:
            dados = response.json()
            total = dados.get("total_detections", 0)
            print(f"Objetos detetados: {total}")
            
            for det in dados.get("detections", []):
                print(f" -> {det['class']} ({det['confidence'] * 100:.1f}%)")
        else:
            print(f"Erro na API: {response.status_code}")

    except Exception as e:
        print(f"Falha na comunicação: {e}")

    # Mostra o vídeo na tela
    cv2.imshow("Camara - YOLO FastAPI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()