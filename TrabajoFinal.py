# Trabajo Machine Learning
# @authors: Vicente Cifre, Mikel Ardanaz
import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# TODO tener todos los modelos en un mismo diccionario llamado modelos con string modeloparametros y modelo
# TODO modularizar
# TODO testear jerarquico


def kmeans(imagen):
    '''
    Realiza un clustering utilizando el algoritmo KMeans implementado en scikit-learn con 5,10 y 17 clusters.
    :return: * modelos, predictions: El modelo ya entrenado y el resultado de la prediccion
    '''
    modelos=[]
    predictions=[]
    for nclusters in (5,10,17):# Yl va de 0 a 16
        kmeans = KMeans(n_clusters=nclusters, random_state=42,n_jobs=-1).fit(imagen)# njobs=1-> Parallel processing
        modelos.append(kmeans)
        predictions.append(kmeans.predict(imagen))# Prescindible (https://stackoverflow.com/questions/25012342/scikit-learns-k-means-what-does-the-predict-method-really-do)
    return modelos, predictions


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
    # 1º Aproximación: Kmeans 5,10,17 clusters
    modelos, predictions= kmeans(imagen)
    for i in range(3):
        ax = plt.subplot(1, 3, i+1)
        ax.imshow(predictions[i].reshape((145, 145), order="F"))
        plt.title('K-means '+str(modelos[i].n_clusters)+' clusters') # Dims(Y)
    plt.show()
    # 2º Aprox: Filter background
    # Probamos ahora a hacer cluster solo de los que tienen etiqueta (al resto les mantendremos el valor de 0)
    Y_final_orig = np.zeros((Yl.shape[0], 1))
    Xl_SoloClasei = np.float64(Xl[Yl != 0, :])
    modelos1 = KMeans(n_clusters=16, random_state=42).fit(Xl_SoloClasei)
    Y_final_orig[Yl != 0, 0] = modelos1.labels_ + 1 # Kmeans empieza a etiquetar en 0
    plt.imshow(Y_final_orig.reshape((145, 145), order="F"))
    plt.title('K-means 16 clusters, solo etiquetados')
    plt.show()
    # 3º Aprox: Repetimos 1º y 2º Aprox. con estandarización
    stander = StandardScaler()
    Xl_std = stander.fit_transform(Xl)
    imagen = np.float64(Xl_std[:, (range(0, 220))])
    modelos, predictions = kmeans(imagen)
    for i in range(3):
        ax = plt.subplot(1, 3, i + 1)
        ax.imshow(predictions[i].reshape((145, 145), order="F"))
        plt.title('K-means std ' + str(modelos[i].n_clusters) + ' clusters')  # Dims(Y)
    plt.show()
    Xl_stdclase = np.float64(Xl_std[Yl != 0, :])
    modelos1 = KMeans(n_clusters=16, random_state=42).fit(Xl_stdclase)
    Y_final_orig[Yl != 0, 0] = modelos1.labels_ + 1  # Kmeans empieza a etiquetar en 0
    plt.imshow(Y_final_orig.reshape((145, 145), order="F"))
    plt.title('K-means std 16 clusters, etiquetados')
    plt.show()

    # Dibujamos las imagenes
    ax=plt.subplot(1,2,1)
    ax.imshow(X[:,:,1]), ax.axis('off'), plt.title('Image')
    ax=plt.subplot(1,2,2)
    ax.imshow(Y), ax.axis('off'), plt.title('Ground Truth')

    # Dibujamos los resultados
    clasmap = Y;  # aqui deberiamos poner nuestra clasificacion
    clasmap_masked = np.ma.masked_where(clasmap < 1, clasmap)
    # for i in range(15):
    # plt.imshow(X[:,:,1])

    plt.imshow(clasmap_masked)
    plt.show()