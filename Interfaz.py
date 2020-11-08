# Se hacen las importaciones necesarias
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os
from shutil import rmtree
import cv2
from PIL import Image, ImageTk
import glob
import imutils
import numpy as np
import time 

'''
En esta parte iran todas las funciones necesarias para que los botones principales de las pestañas puedan funcionar
'''
# Se crea una funcion para obtener el directorio donde estan las personas
def directorio():
    obtener_directorio = os.getcwd() # Obtiene el directorio donde esta el programa
    directorio = obtener_directorio + "/Personas" # Cambia al directorio final
    return directorio

# Se crean variables para mostrar la camara
width, height = 400, 300
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Se crea una funcion para visualizar la camara
def show_frame():
       _, frame = cap.read()
       frame = cv2.flip(frame, 1)
       cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
       img = Image.fromarray(cv2image)
       imgtk = ImageTk.PhotoImage(image=img)
       lmain.imgtk = imgtk
       lmain.configure(image=imgtk)
       lmain.after(10, show_frame)

#Boton que inicia el programa de identificacion de rostros
def iniciarPrograma():
    direccionCarpeta = directorio()
    imagePaths = os.listdir(direccionCarpeta)
    print('imagePaths = ',imagePaths)

    reconocimiento_rostro = cv2.face.EigenFaceRecognizer_create()

    #Leyendo modelo
    reconocimiento_rostro.read('modeloEigenFace.xml')

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    clasificadorRostros = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if ret == False: break
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gris.copy()

        caras = clasificadorRostros.detectMultiScale(gris,1.3,5)

        for (x,y,w,h) in caras:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(300,300),interpolation=cv2.INTER_CUBIC)
            resultado = reconocimiento_rostro.predict(rostro)

            cv2.putText(frame, '{}'.format(resultado), (x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

            # eifenfaces
            if resultado[1] < 8500:
                cv2.putText(frame, '{}'.format(imagePaths[resultado[0]]), (x,y-25),1,1.3,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            else:
                personaDesconocida = frame.copy()
                carpetaDesconocido = direccionCarpeta + '/Desconocido'
                if not os.path.exists(carpetaDesconocido):
                    print('Carpeta creada: ', carpetaDesconocido)
                    os.makedirs(carpetaDesconocido)
                
                contador = len(glob.glob(carpetaDesconocido + "/*.jpg"))
                
                cv2.imwrite(carpetaDesconocido + '/rostro_{}.jpg'.format(contador),personaDesconocida)

                cv2.putText(frame, 'Desconocido', (x,y-20),1,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows() 

# Se crea una funcion para ingresar los datos de una persona nueva
def agregar():
    # Variables Globales
    global lmain
    global camara_interfaz
    camara_interfaz = False
    '''
    Espacio para las funciones de la pestaña emergente
    '''
    # Funcion para guardar y capturar rostros
    def capturarRostro():
        direccionCarpeta = directorio()
        carpetaPersona = direccionCarpeta + '/' + nombre.get()
        if not os.path.exists(carpetaPersona):
            print('Carpeta creada: ', carpetaPersona)
            os.makedirs(carpetaPersona)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        clasificadorRostros = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        contador = len(glob.glob(carpetaPersona + "/*.jpg"))
        final = contador + 300
        while True:
            ret, frame = cap.read()
            if ret == False: break
            frame = imutils.resize(frame, width=640)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()

            caras = clasificadorRostros.detectMultiScale(gris,1.3,5)

            for (x,y,w,h) in caras:
                cv2.rectangle(frame, (x,y), (x+w,+h), (255,0,0), 2)
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(300,300),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(carpetaPersona + '/rostro_{}.jpg'.format(contador),rostro)
                contador = contador + 1
            cv2.imshow('frame', frame)    

            k = cv2.waitKey(1)
            if k == 27 or contador >= final:
                break

        cap.release()
        cv2.destroyAllWindows()

    # Funcion para entrenar a la red
    def entrenar():
        direccionCarpeta = directorio()
        listaPersonas = os.listdir(direccionCarpeta)

        labels = []
        facesData = []
        label = 0

        for nombreDir in listaPersonas:
            personPath = direccionCarpeta + '/' + nombreDir
            print('Leyendo las iamgenes')

            for nombreArchivo in os.listdir(personPath):
                print('Rostros: ', nombreDir + '/' + nombreArchivo)
                labels.append(label)
                facesData.append(cv2.imread(personPath+'/'+ nombreArchivo,0))
                imagen = cv2.imread(personPath+'/'+ nombreArchivo,0)

            label = label + 1

        reconocimiento_rostro = cv2.face.EigenFaceRecognizer_create()

        # Entrenamiento reconocedor de rostros
        print('Entrenando...')    
        reconocimiento_rostro.train(facesData,np.array(labels))

        # Almacenar el entrenamiento
        reconocimiento_rostro.write('modeloEigenFace.xml')
        print('Modelo almacenado...')

    # Se crear una nueva ventana 
    formulario = Toplevel() 
    formulario.title("Agregar usuario nuevo")
    formulario.config(bg = "#A8CBEF")
    # Se agregan los label y la entrada de texto
    texto = Label(formulario, text = "Ingrese los datos", bg = "#A8CBEF").pack(pady=15)
    ingrese_nombre = Label(formulario, text = "Ingrese el nombre", bg = "#A8CBEF").pack()
    nombre = Entry(formulario, width = 30, borderwidth=5)
    nombre.pack()
    # Se agrega la webcam
    lmain = Label(formulario)
    lmain.pack(pady=10)
    
    # Se crean los botones
    buton_guardar = Button(formulario,text = "Capturar Rostro", bg = "#A2ABB5", borderwidth=4, width = 20, height = 1, command = capturarRostro).pack()
    buton_entrenar = Button(formulario,text = "Actualizar Datos", bg = "#A2ABB5", borderwidth=4, width = 20, height = 1, command = entrenar).pack(pady=10)
    if (camara_interfaz == True):
        show_frame()

# Se crea funcion para eliminar carpetas
def eliminar():
    direccionCarpeta = directorio()
    # Codigo que elimina la carpeta que ha sido seleccionada en el cuadro de dialogo
    usuario = filedialog.askdirectory(initialdir=direccionCarpeta, title="Selecciona una carpeta")
    carpeta = usuario
    rmtree(carpeta)
  

'''
En esta parte esta la parte de la interfaz
'''
# Se crea la ventana 
ventana = Tk()
ventana.title("Identificador de Rostros")
ventana.geometry('300x200')

# Se crean las pestañas
tab_control = ttk.Notebook(ventana)
tab_control.pack(fill='both', expand=1)

inicio = Frame(tab_control, bg = "#A8CBEF", bd=12, relief="sunken")
editar = Frame(tab_control, bg = "#A8CBEF", bd=12, relief="sunken")

inicio.pack(fill='both', expand=1)
editar.pack(fill='both', expand=1)

tab_control.add(inicio, text = "Inicio")
tab_control.add(editar, text = "Editar")

# Se crean los botones para la pestaña Inicio
boton_iniciar = Button(inicio, text = "Iniciar Programa", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1, command=iniciarPrograma)
boton_iniciar.pack(pady=15)

boton_cerrar = Button(inicio, text = "Cerrar Programa", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1,command=ventana.quit)
boton_cerrar.pack(pady=15)      

# Se crean los botones para la pestaña Editar
boton_agregar = Button(editar, text = "Agregar", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1,command = agregar)
boton_agregar.pack(pady=15)

boton_eliminar = Button(editar, text = "Eliminar", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1,command = eliminar)
boton_eliminar.pack(pady=15)

ventana.mainloop()
