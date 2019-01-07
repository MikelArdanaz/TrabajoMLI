# Trabajo Machine Learning
# @authors: Vicente Cifre, Mikel Ardanaz
import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# TODO tener todos los modelos en un mismo diccionario llamado modelos con string modeloparametros y modelo
# TODO testear jerarquico


def kmeans(imagen,tipo='',labeled=False):
    '''
    Realiza un clustering utilizando el algoritmo KMeans implementado en scikit-learn con 5,10 y 17 clusters.
    :return: * modelos, predictions: El modelo ya entrenado y el resultado de la prediccion
    '''
    modelos=[]
    predictions=[]
    for i,nclusters in enumerate([5,10,17]):# Yl va de 0 a 16
        kmeans = KMeans(n_clusters=nclusters, random_state=42,n_jobs=-1).fit(imagen)# njobs=1-> Parallel processing
        modelos.append(kmeans)
        predictions.append(kmeans.predict(imagen))# Prescindible (https://stackoverflow.com/questions/25012342/scikit-learns-k-means-what-does-the-predict-method-really-do)
        ax = plt.subplot(1, 3, i+1)
        if labeled:
            plt.suptitle('Kmeans etiquetado')
            Y_final_orig[Yl != 0, 0] = modelos[i].labels_ + 1
            predictions[i]=Y_final_orig
        else:
            plt.suptitle('Kmeans')
        ax.imshow(predictions[i].reshape((145, 145), order="F"))
        plt.title(tipo+str(modelos[i].n_clusters)+' clusters') # Dims(Y)
    plt.show()
    return modelos, predictions
def PCApply(X):
    data=PCA(n_components=40).fit_transform(X)
    _, b,_ = np.linalg.svd(X.transpose().dot(X))# Demo mejor con 40
    plt.title('Explicación variabilidad en base al número de variables')
    plt.plot(range(10, 75), b[10:75], 'bx-')
    plt.show()
    return data
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
    modelos, predictions = kmeans(imagen,'std')
    Xl_stdclase = np.float64(Xl_std[Yl != 0, :])
    modelos1 = KMeans(n_clusters=16, random_state=42).fit(Xl_stdclase)
    Y_final_orig[Yl != 0, 0] = modelos1.labels_ + 1  # Kmeans empieza a etiquetar en 0
    plt.imshow(Y_final_orig.reshape((145, 145), order="F"))
    plt.title('K-means std 16 clusters, etiquetados')
    plt.show()
    #4º Aprox: PCA
    data=PCApply(Xl_std)
    modelosPCA, predictionsPCA=kmeans(data[Yl!=0,:],'PCA + std ',labeled=True)
    # 5º Aprox: Gaussian Mixtures
    GM = mixture.GaussianMixture(n_components=16, random_state=42).fit_predict(data[Yl!=0,:])
    Y_GM=np.zeros((Yl.shape[0], 1))
    Y_GM[Yl != 0, 0] = GM + 1
    plt.imshow(Y_GM.reshape((145, 145), order="F"))
    plt.title('Gaussian Mixtures, std + PCA 16 clusters, etiquetados')
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