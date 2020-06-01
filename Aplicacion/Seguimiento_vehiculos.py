from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import tkinter
import cv2
import numpy as np
from time import time
import time
import pytesseract
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.models import Sequential, load_model
import os
#import cca2
# Para lecturas de matrículas en window
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#Precisión de que sea matrícula
CONF_THRESH, NMS_THRESH = 0.2, 0.2

# yolov3_b.cfg #'darknet/yolov31.cfg'
CONFIG = "src//yolov3_final.cfg"
# yolov3_bn.weights #'darknet/backup/yolov3_2000.weights'
WEIGHTS = "src//yolov3_last.weights"
NAMES = "src//obj.names"  # 'darknet/obj.names'


def canvasInterfaceframeFotos(frame):
    canvasCoches = tkinter.Canvas(frame, width=800, height=400, bg="black")
    canvasCoches.grid(row=0, column=0)
    label1 = tkinter.Label(frame, text="Matrículas Encontradas")
    label1.grid(row=0, column=1, padx=10)
    lista = tkinter.Listbox(frame)
    lista.grid(row=0, column=2, padx=10)
    runButton = tkinter.Button(frame, state=DISABLED, text="Buscar Matrículas", command=lambda: ejecutar_red(
        ruta_ima, canvasCoches, lista))  
    runButton.grid(row=1, column=1, pady=10)
    insertButton = tkinter.Button(
        frame, text="Insertar", command=lambda: abrirArchivo(canvasCoches, runButton))
    insertButton.grid(row=1, column=0, pady=10)


def CurSelet(evt):
    ima = cv2.imread("src//coche.jpg")
    tex = lista.get(lista.curselection())
    cv2.putText(ima, tex, (170, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imwrite("src//coche_ma.jpg", ima)
    global imagenb
    im = Image.open("src//coche_ma.jpg")
    o_size = im.size
    f_size = (400, 200)
    factor = min(float(f_size[1])/o_size[1], float(f_size[0])/o_size[0])
    width = int(o_size[0] * factor)
    height = int(o_size[1] * factor)

    rImg = im.resize((width, height), Image.ANTIALIAS)
    imagenb = ImageTk.PhotoImage(rImg)
    labelFoto.configure(image=imagenb)


def canvasInterfaceframeVideos(frame):
    label1 = tkinter.Label(frame, text="Matriculas Encontradas:")
    label1.grid(row=2, column=2, ipady=40)
    global lista
    global labelFoto
    lista = tkinter.Listbox(frame)
    lista.grid(row=2, column=3, pady=10)
    lista.bind('<<ListboxSelect>>', CurSelet)
    verButton = tkinter.Button(
        frame, state=DISABLED, text="Ver Video", command=lambda: verVideo("video_procesado.avi"))
    verButton.grid(row=3, column=1, pady=60)
    runButton = tkinter.Button(frame, state=DISABLED, text="Buscar Matrículas", command=lambda: ejecutar_red_2(
        verButton, ruta_ima, lista))  # ejecutar_red(ruta_ima, canvasCoches, lista))
    runButton.grid(row=2, column=1)
    insertButton = tkinter.Button(
        frame,  text="Insertar Vídeo", command=lambda: abrirArchivoVideo(runButton))
    insertButton.grid(row=1, column=1, pady=60, padx=150, ipady=10, sticky="s")
    ima = cv2.imread("src//coche.jpg")
    cv2.putText(ima, "Selecciona la matricula", (140, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite("src//coche_ma.jpg", ima)
    global imagena
    im = Image.open("src//coche_ma.jpg")
    o_size = im.size
    f_size = (400, 200)
    factor = min(float(f_size[1])/o_size[1], float(f_size[0])/o_size[0])
    width = int(o_size[0] * factor)
    height = int(o_size[1] * factor)

    rImg = im.resize((width, height), Image.ANTIALIAS)
    imagena = ImageTk.PhotoImage(rImg)
    labelFoto = tkinter.Label(frame, image=imagena)
    labelFoto.grid(row=2, column=4, padx=50, sticky=S+N+E+W)


def abrirVentana():

    ventanaProgreso = tkinter.Toplevel(ventanaVideos)
    label1 = tkinter.Label(
        ventanaProgreso, text="Procesamiento del video finalizado")
    label1.pack(ipadx=10, ipady=10)


def iniciarVentanas(ventana):
    ventana.geometry("1280x600+50+50")
    ventana.minsize(1280, 600)


def abrirArchivo(canvasCoches, runButton):
    global ruta_ima
    runButton['state'] = 'normal'
    archivo = filedialog.askopenfile()
    ruta_ima = archivo.name
    global rImg
    im = Image.open(archivo.name)
    o_size = im.size
    f_size = (800, 400)
    factor = min(float(f_size[1])/o_size[1], float(f_size[0])/o_size[0])
    width = int(o_size[0] * factor)
    height = int(o_size[1] * factor)
    rImg = im.resize((width, height), Image.ANTIALIAS)
    rImg = ImageTk.PhotoImage(rImg)
    canvasCoches.create_image(
        f_size[0]/2, f_size[1]/2, image=rImg, anchor=tkinter.CENTER)


def abrirArchivoVideo(runButton):
    global ruta_ima
    runButton['state'] = 'normal'
    archivo = filedialog.askopenfile()
    ruta_ima = archivo.name



def verVideo(ruta_video):
    video = cv2.VideoCapture(ruta_video)
    while(video.isOpened()):
        ret, frame_image = video.read()
        if ret:
            cv2.namedWindow('tk', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('tk', 800, 400)
            cv2.imshow('tk', frame_image)
            #prueba = ImageTk.PhotoImage(image = Image.fromarray(frame))
            cv2.waitKey(25)
        else:
            break

    video.release()

def mostrar_imagen(canvasCoches):
    global rImg
    im = Image.open("resultado"+str(nombre_img)+".jpg")
    o_size = im.size
    f_size = (800, 400)
    factor = min(float(f_size[1])/o_size[1], float(f_size[0])/o_size[0])
    width = int(o_size[0] * factor)
    height = int(o_size[1] * factor)
    rImg = im.resize((width, height), Image.ANTIALIAS)
    rImg = ImageTk.PhotoImage(rImg)
    canvasCoches.create_image(
        f_size[0]/2, f_size[1]/2, image=rImg, anchor=tkinter.CENTER)


def inicalizacion():
    frameInicio = tkinter.Frame(ventanaPrincipal)
    frameInicio.pack(fill='both', expand=1, padx=400)
    buttonFoto = tkinter.Button(frameInicio, text="Fotos", command=pulsaFotos)
    buttonFoto.pack(ipadx=10, ipady=10, side=LEFT, padx=100)
    buttonVideo = tkinter.Button(
        frameInicio, text="Video", command=pulsaVideos)
    buttonVideo.pack(side=LEFT, ipadx=10, ipady=10)

def pulsaFotos():
    global fotos
    fotos = True
    ventanaVideos.deiconify()
    ventanaPrincipal.withdraw()
    frameCanvas = tkinter.Frame(ventanaVideos)
    frameCanvas.columnconfigure(0, weight=1)
    frameCanvas.rowconfigure(0, weight=1)
    frameCanvas.grid(sticky="nsew")
    canvasInterfaceframeFotos(frameCanvas)


def pulsaVideos():
    global fotos
    fotos = False
    ventanaVideos.deiconify()
    ventanaPrincipal.withdraw()
    global frameCanvas
    frameCanvas = tkinter.Frame(ventanaVideos)
    frameCanvas.columnconfigure(0, weight=1)
    frameCanvas.rowconfigure(0, weight=1)
    frameCanvas.grid(sticky="nsew")
    canvasInterfaceframeVideos(frameCanvas)


def cerrar_app():
    ventanaPrincipal.destroy()


def ejecutar_red_2(verButton, imagen_red, lista):
    lista.delete(0, END)
    global frame
    vs = cv2.VideoCapture(imagen_red)
    writer = None
    (W, H) = (None, None)

    length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Tiempo aproximado: ", length/60, "min")
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)

        layerOutputs = net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > CONF_THRESH:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH,
                                CONF_THRESH)


        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                crop_img = frame[y:y+h, x:x+w]
                texto = pytesseract.image_to_string(
                    crop_img, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7') 

                numeros = "0123456789"
                if (len(texto) > 6 and len(texto) < 8):
                    if (texto[0] in numeros and texto[1] in numeros and texto[2] in numeros and texto[3] in numeros):
                        print(texto)
                        dicc.setdefault(texto)
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(texto,
                                           confidences[i])

                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            writer = cv2.VideoWriter("video_procesado.avi", fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)

    # release the file pointers
    print("Proceso Terminado")
    writer.release()
    vs.release()
    verButton['state'] = 'normal'
    abrirVentana()
    for key in dicc:
        lista.insert(tkinter.END, key)


def ejecutar_red(imagen_red, canvasCoches, lista):
    lista.delete(0, END)
    img = cv2.imread(imagen_red)
    img_bn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img_bn.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (
                    detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    indices = cv2.dnn.NMSBoxes(
        b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

    # Draw the filtered bounding boxes with their class to the image
    with open(NAMES, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    color = (255, 0, 0)
    color1 = (0, 0, 255)
    #current_dir = os.path.dirname(os.path.realpath(__file__))
    #model_dir = os.path.join(current_dir, 'models/svc/SVC_model.pkl')
    #model1 = load_model('character_recognition.h5')
    #model1 = load_model('cnn_classifier.h5')
    #model = joblib.load(model_dir)

    for index in indices:
        charList = []
        x, y, w, h = b_boxes[index]
        print(x, y, w, h)
        crop_img = img_bn[y:y+h, x:x+w]
        crop_img2 = img[y:y+h, x:x+w]
        """
        result = np.zeros(crop_img2.shape, dtype=np.uint8)
        hsv = cv2.cvtColor(crop_img2, cv2.COLOR_BGR2HSV)
        lower = np.array([0,0,0])
        upper = np.array([179,100,130])
        mask = cv2.inRange(hsv, lower, upper)

        # Perform morph close and merge for 3-channel ROI extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        extract = cv2.merge([close,close,close])

        # Find contours, filter using contour area, and extract using Numpy slicing
        cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = w * h
            if area < 5000 and area > 2500:
                cv2.rectangle(crop_img2, (x, y), (x + w, y + h), (36,255,12), 3)
                result[y:y+h, x:x+w] = extract[y:y+h, x:x+w] 

        # Invert image and throw into Pytesseract
        invert = 255 - result
        data = pytesseract.image_to_string(invert, lang='eng',config='--psm 6')

        """
        texto = pytesseract.image_to_string(
            crop_img2, config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7')  # config='--psm 7'
        print(texto)
        if (len(texto) > 5 ):
            lista.insert(lista.size(), texto)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(texto,confidences[index])
        cv2.putText(img, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)

    global nombre_img
    nombre_img = nombre_img + 1
    cv2.imwrite("resultado"+str(nombre_img)+".jpg", img)
    mostrar_imagen(canvasCoches)


# MAIN

tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# Inicializacion parametros de la red
LABELS = open(NAMES).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) # GPU: cv2.dnn.DNN_BACKEND_CUDA #cv2.dnn.DNN_BACKEND_OPENCV
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # GPU: cv2.dnn.DNN_TARGET_CUDA #cv2.dnn.DNN_TARGET_CPU
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]


dicc = {}
nombre_img = 0
ventanaPrincipal = tkinter.Tk()
ventanaVideos = tkinter.Toplevel(ventanaPrincipal)
ventanaVideos.protocol("WM_DELETE_WINDOW", cerrar_app)
iniciarVentanas(ventanaPrincipal)
iniciarVentanas(ventanaVideos)
ventanaVideos.withdraw()

imagenes = []

inicalizacion()

ventanaPrincipal.mainloop()
