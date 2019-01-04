
# coding: utf-8

import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt


# Lectura de la imagen de fichero de Matlab .mat
mat_file =  "datasetB1.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) #devuelve un dictionary
list(mat.keys()) #variables almacenadas


# Lectura de los datos
X = mat["X"]   #imagen (hipercubo 3D: filas x columnas x variables) 
Xl = mat["Xl"]   #muestras etiquetadas (muestas x variables) 
Yl = mat["Yl"]   #etiquetas de clases (muestras x 1, 0=sin clase)   
del mat
Yl.shape


# Reshape del Ground Truth como una imagen
Y = np.reshape(Yl, (X.shape[0], X.shape[1]),order="F")
Y.shape


# Filter background: eliminamos la clase 0 de los datos etiquetados
Nc=Yl.max()-Yl.min()+1
if Nc>2:
    Xl = Xl[Yl != 0,:];
    Yl = Yl[Yl != 0];


# Dibujamos las imagenes
ax=plt.subplot(1,2,1)
ax.imshow(X[:,:,1]), ax.axis('off'), plt.title('Image')
ax=plt.subplot(1,2,2)
ax.imshow(Y), ax.axis('off'), plt.title('Ground Truth')


# Dibujamos los resultados
clasmap=Y; #aqui deberiamos poner nuestra clasificacion
clasmap_masked = np.ma.masked_where(clasmap<1,clasmap)
plt.imshow(X[:,:,1])
plt.imshow(clasmap_masked)

