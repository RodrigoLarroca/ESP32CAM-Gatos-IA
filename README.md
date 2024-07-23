# ESP32CAM-Gatos-IA
## Deteccion de gatos con sonido con esp32cam.

![image](https://github.com/user-attachments/assets/bc02647e-edca-431d-bd88-fe3cdea6e3aa)
![image](https://github.com/user-attachments/assets/4c3b8863-5632-4df1-90cf-b62696dc34fb)

Pasos:
1- pip install pyinstaller

2- pyinstaller --onefile --windowed detector_gatos.py

3- Después de ejecutar el comando anterior, PyInstaller creará una carpeta dist en tu directorio actual, y dentro de ella estará tu archivo ejecutable. En esa carpeta es necesario tener los siguientes archivos:

* coco.names
* yolov3
* yolov3.weights
* yolov3-tiny
* yolov3-tiny.weights

