from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os
from shutil import rmtree
import cv2
from PIL import Image, ImageTk

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
boton_iniciar = Button(inicio, text = "Iniciar Programa", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1)
boton_iniciar.pack(pady=15)

boton_cerrar = Button(inicio, text = "Cerrar Programa", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1,command=ventana.quit)
boton_cerrar.pack(pady=15)

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

# Se crea una funcion para ingresar los datos de una persona nueva
def agregar():
    global lmain
    formulario = Toplevel() # Se crear una nueva 
    formulario.title("Agregar usuario nuevo")
    formulario.config(bg = "#A8CBEF")
    texto = Label(formulario, text = "Ingrese los datos", bg = "#A8CBEF").pack(pady=15)
    nombre = Entry(formulario, width=30, borderwidth=5).pack()
    lmain = Label(formulario)
    lmain.pack(pady=10)
    buton_guardar = Button(formulario,text = "Capturar Rostro", bg = "#A2ABB5", borderwidth=4, width = 20, height = 1).pack()
    buton_entrenar = Button(formulario,text = "Actualizar Datos", bg = "#A2ABB5", borderwidth=4, width = 20, height = 1).pack(pady=10)
    show_frame()
    

# Se crea funcion para eliminar carpetas
def eliminar():
    directorio = os.getcwd() # Obtiene el directorio donde esta el programa
    directorio_nuevo = directorio.replace("\ ", "/") # Cambia las diagonale para que python las pueda leer
    directorio_final = directorio + "/Usuarios" # Cambia al directorio final
    # Codigo que elimina la carpeta que ha sido seleccionada en el cuadro de dialogo
    usuario = filedialog.askdirectory(initialdir=directorio_final, title="Selecciona una carpeta")
    carpeta = usuario
    rmtree(carpeta)
   

# Se crean los botones para la pestaña Editar
boton_agregar = Button(editar, text = "Agregar Usuario", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1,command = agregar)
boton_agregar.pack(pady=15)

boton_eliminar = Button(editar, text = "Eliminar Usuario", bg = "#A2ABB5", borderwidth=4, width = 15, height = 1,command = eliminar)
boton_eliminar.pack(pady=15)

ventana.mainloop()