'''
Trabajo Machine Learning
@authors: Vicente Cifre, Mikel Ardanaz
'''
import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# TODO tener todos los modelos en un mismo diccionario llamado modelos con string modeloparametros y modelo

def Kmeans(imagen):
    '''
    Realiza un clustering utilizando el algoritmo KMeans implementado en scikit-learn con 5,10y 20 clusters.
    :return: * modelos, predictions: El modelo ya entrenado y el resultado de la prediccion
    '''
    modelos=[]
    predictions=[]
    for nclusters in (5,10,20):
        kmeans = KMeans(n_clusters=nclusters, random_state=42).fit(imagen)
        modelos.append(kmeans)
        predictions.append(kmeans.predict(imagen))
    return modelos,predictions


if __name__ == '__main__':
    # Lectura de la imagen de fichero de Matlab .mat
    mat_file ="datasetB1.mat"
    mat = matlab.loadmat(mat_file,squeeze_me=True) #devuelve un dictionary
    list(mat.keys()) #variables almacenadas


    # Lectura de los datos
    X = mat["X"]   #imagen (hipercubo 3D: filas x columnas x variables)
    Xl = mat["Xl"]   #muestras etiquetadas (muestas x variables)
    Yl = mat["Yl"]   #etiquetas de clases (muestras x 1, 0=sin clase)
    del mat


    # Reshape del Ground Truth como una imagen
    Y = np.reshape(Yl, (X.shape[0], X.shape[1]),order="F")
    imagen=np.float64(Xl[:,(range(0,220))])#imagen en sus 220 dimensiones
    modelos,predictions=Kmeans(imagen)
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(predictions[i].reshape((145, 145), order="F")), plt.title('K-means clustering ')

    # Filter background: eliminamos la clase 0 de los datos etiquetados
    # Nc=Yl.max()-Yl.min()+1
    # if Nc>2:
    #     Xl = Xl[Yl != 0,:];
    #     Yl = Yl[Yl != 0];


    # Dibujamos las imagenes
    ax=plt.subplot(1,2,1)
    ax.imshow(X[:,:,1]), ax.axis('off'), plt.title('Image')
    ax=plt.subplot(1,2,2)
    ax.imshow(Y), ax.axis('off'), plt.title('Ground Truth')
