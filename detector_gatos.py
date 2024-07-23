import cv2
import numpy as np
import winsound

# URL del stream de la ESP32-CAM
url = 'http://xxx.xxx.x.xx:xx/stream'

# Función para ajustar el brillo
def ajustar_brillo(imagen, brillo=30):
    return cv2.convertScaleAbs(imagen, alpha=1, beta=brillo)

# Cargar modelo de detección de objetos (YOLO)
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()

# Asegurarse de que estamos obteniendo los índices de las capas de salida correctamente
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Clases de objetos que YOLO puede detectar
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Iniciar captura de video
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reducir la resolución del video
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede recibir el frame (stream terminado?). Saliendo ...")
            break

        height, width, channels = frame.shape

        # Ajustar el brillo del frame
        frame = ajustar_brillo(frame, brillo=50)

        # Preprocesamiento para YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # Reducir el tamaño del blob
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Análisis de las salidas de YOLO
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "cat":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Alerta sonora al detectar un gato
                    winsound.Beep(1000, 500)  # Frecuencia de 1000 Hz y duración de 500 ms

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Dibujar las cajas en la imagen
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow('ESP32-CAM', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # Aumentar el tiempo de espera
            break

except KeyboardInterrupt:
    print("Interrupción manual detectada. Cerrando...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados y aplicación cerrada correctamente.")
