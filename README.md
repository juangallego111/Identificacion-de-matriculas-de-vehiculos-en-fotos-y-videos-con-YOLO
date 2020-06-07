# Identificacion-de-matriculas-de-vehiculos-en-fotos-y-videos-con-YOLO
TFG - SEGUIMIENTO DE VEHÍCULOS EN SECUENCIAS DE VÍDEO A PARTIR DE IDENTIFICACIÓN DE MATRÍCULAS

Sistema que facilita el seguimiento de vehículos presentes en secuencias de vídeo, partiendo de la
identificación de sus placas de matrícula a partir de la red neuronal YOLOv3.

## Contenido
-Red neuronal YOLO : aquí se encuentra la implementación con YOLO de la red neuronal. Dentro de esta carpeta nos encontramos con:
  -> pre-entrenamiento: scripts que facilitan crear la base de datos de la red neuronal con los formatos adecuados.
  -> darknet: en esta carpeta se encuentra la configuración, imágenes para el entrenamiento y evaluación de red. Con esta carpeta se puede entrenar nuevamente la red.
- Aplicación: interfaz para la identificación de los vehículos. Dentro de la carpeta se encuentran capturas de las fotos e imágenes que obtenemos como resultado (incluidos en la carpeta 'test'). También se incluyen en la carpeta 'src' los archivos necesarios para el correcto funcionamiento de la aplicación (pesos y archivos de configuración).

## Instalación
La instalación de la aplicación se realiza instalando las librerias incluidas en el archivo requirements.txt. Para instarlas todas se puede ejecutar la siguiente orden:
```bash
pip install -r requirements.txt
```
El directorio src debe estar en la misma ubicación que el archivo python "Seguimiento_vehiculos.py".
Finalmente podemos ejecutar la aplicación desde la terminal con la orden:
```bash
python Seguimiento_vehículos.py
```
## Capturas


![](/Aplicacion/Interfaz_fotos.PNG)
![](/Aplicacion/interfaz_videos.PNG)
