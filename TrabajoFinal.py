# Trabajo Machine Learning
# @authors: Vicente Cifre, Mikel Ardanaz
import numpy as np
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, adjusted_rand_score, mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

# TODO testear jerarquico


def kmeans(imagen,tipo='',labeled=False):
    '''
    Realiza un clustering utilizando el algoritmo KMeans implementado en scikit-learn con 5,10 y 17 clusters.
    Posteriormente realiza una gráfica con los resultados.
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


def pcapply(X):
    '''
    Realiza un PCA y muestra en una gráfica la explicación de la variabilidad respecto al nº de vars. La hemos usado
    para elegir el número de variables que queriamos conservar.
    :param X:Dataset
    :return: data -  Conjunto de datos reducido
    '''
    data=PCA(n_components=40).fit_transform(X)
    _, b,_ = np.linalg.svd(X.transpose().dot(X))# Demo mejor con 40
    plt.title('Explicación variabilidad en base al número de variables')
    plt.plot(range(10, 75), b[10:75], 'bx-')
    plt.show()
    return data


def plotmetrics(Yl,modelos):
    '''
    Muestra en una gráfica algunas de los medidas de bondad estudiadas
    :param Yl: Etiquetas de clases
    :param modelos: Diccionario con NombreModelo y clases asignadas
    :return:
    '''
    mutualinfo={}
    vmeasure={}
    rand={}
    for i in modelos:
        mutualinfo[i]=mutual_info_score(Yl[Yl!=0], modelos[i])
        vmeasure[i]=v_measure_score(Yl[Yl!=0], modelos[i])
        rand[i]=adjusted_rand_score(Yl[Yl!=0], modelos[i])
    mutualinfo['Ground Truth']=mutual_info_score(Yl[Yl!=0],Yl[Yl!=0])
    vmeasure['Ground Truth']=v_measure_score(Yl[Yl!=0],Yl[Yl!=0])
    rand['Ground Truth']=adjusted_rand_score(Yl[Yl!=0],Yl[Yl!=0])
    plt.subplot(221)
    plt.bar(range(len(mutualinfo)), mutualinfo.values(), align='center')
    plt.title('Información Mutua')
    plt.xticks(range(len(mutualinfo)), mutualinfo.keys())
    plt.subplot(222)
    plt.bar(range(len(vmeasure)), vmeasure.values(), align='center')
    plt.title('V Measure')
    plt.xticks(range(len(vmeasure)), vmeasure.keys())
    plt.subplot(223)
    plt.bar(range(len(rand)), rand.values(), align='center')
    plt.title('Rand')
    plt.xticks(range(len(rand)), rand.keys())
    plt.show()
    # TODO no se ve bien a que corresponde cada barra


def elbow(Xl,Yl):
    '''
    Implementación del método Elbow (https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set)
    para conocer el número de clusters que debe tener cada dataset.Lo usaremos posteriormente en la selección de muestras.
    El número de clusters se elige a ojo en función de donde encontremos el codo y del tiempo necesario para el aprendizaje.
    IMP: En algunos casos no esta bien definido el codo. (¡La vida real es asi de dura!)
    '''
    for clase in range(1,17):
        indexclasified=np.where(Yl ==clase)[0]# Indexes of class
        Xlclase=Xl[indexclasified,:]
        model=KMeans()
        visualizer = KElbowVisualizer(model, k=(1, 12),title=('Método Elbow para la clase ')+str(clase))
        visualizer.fit(Xlclase)
        visualizer.poof()


def seleccionPuntos(clasificacion,total=5000):
    '''
    :param clasificacion: Criterio usado para para la selección de elementos
    :param total: Número de elementos a particionar
    Implementación proporcional; Mismo número de elementos de cada clase. Rara vez tendremos 5000, lo habitual es tener alguno menos.
    :return: npuntos -- puntos por clase/cluster
    '''
    npuntos=[]
    for clase in np.unique(clasificacion):
        npuntos.append(int(total * np.sum(clasificacion == clase) / clasificacion.shape[0]))#Resultados truncados (No podemos tener medio dato)
    return npuntos


def muestreo(Xl,Yl,Nclusters):
    '''
    Implementa la mixtura de gaussianos ya que es el clustering que mejores resultados nos ha dado.
    Selecciona ptos. + cerca de las medias
    :param Xl: Muestras etiquetadas
    :param Yl: Etiquetas de clases
    :param Nclusters: lista con nº clusters por clase (Obtenidos vía elbow)
    :return: Índices de los puntos representantes
    '''
    # TODO gráfica clasificación
    ptsxclase=seleccionPuntos(Yl[Yl>0])# Puntos para cada clase
    Yl_final = np.zeros(Yl.shape[0])
    for clase in np.unique(Yl[Yl>0]):
        indexclasified=np.where(Yl ==clase)[0]
        GM = mixture.GaussianMixture(n_components=Nclusters[clase - 1]).fit(Xl[indexclasified,:])
        predictions=GM.predict(Xl[indexclasified,:])
        ptsxcluster=seleccionPuntos(predictions,total=ptsxclase[clase-1])
        Yl_Cluster=np.zeros(Yl.shape[0])
        for i, centro in enumerate(GM.means_):
            indexofcluster=indexclasified[predictions==i]# Index of samples belonging to cluster
            Xl_Cluster = Xl[indexofcluster, :]# Cogemos Xl
            norma = np.sqrt(np.sum((Xl_Cluster - centro) ** 2, axis=1))# Diferencia punto con centro
            nearestindexes = norma.argsort()[:ptsxcluster[i]] #Nos quedamos con ptsxcluster elementos
            Yl_Cluster[indexofcluster[nearestindexes]]=i+1
        Yl_final[Yl_Cluster>0]=clase
    return np.where(Yl_final > 0)[0]


if __name__ == '__main__':
    plt.style.use('default')# Al importar yellowbrick, se cambia el esquema de colores a grayscale.
    # Lectura de la imagen de fichero de Matlab .mat
    mat_file ="datasetB1.mat"
    mat = matlab.loadmat(mat_file,squeeze_me=True) #devuelve un dictionary
    list(mat.keys()) #variables almacenadas
    # Lectura de los datos
    X = mat["X"]   #imagen (hipercubo 3D: filas x columnas x variables)
    Xl = mat["Xl"]   #muestras etiquetadas (muestas x variables)
    Yl = mat["Yl"]   #etiquetas de clases (muestras x 1, 0=sin clase)
    del mat
    modelos={}#Dict para almacenar los modelos etiquetados

    # Reshape del Ground Truth como una imagen
    Y = np.reshape(Yl, (X.shape[0], X.shape[1]),order="F")
    imagen=np.float64(Xl[:,(range(0,220))])#imagen en sus 220 dimensiones
    # 1º Aproximación: Kmeans 5,10,17 clusters
    modelo, predictions= kmeans(imagen)
    # 2º Aprox: Filter background
    # Probamos ahora a hacer cluster solo de los que tienen etiqueta (al resto les mantendremos el valor de 0)
    Y_final_orig = np.zeros((Yl.shape[0], 1))
    Xl_SoloClasei = np.float64(Xl[Yl != 0, :])
    modelos1 = KMeans(n_clusters=16, random_state=42).fit(Xl_SoloClasei)
    modelos['Kmeans 16 label'] = modelos1.labels_ + 1
    Y_final_orig[Yl != 0, 0] = modelos1.labels_ + 1 # Kmeans empieza a etiquetar en 0
    plt.imshow(Y_final_orig.reshape((145, 145), order="F"))
    plt.title('K-means 16 clusters, solo etiquetados')
    plt.show()
    # 3º Aprox: Repetimos 1º y 2º Aprox. con estandarización
    stander = StandardScaler()
    Xl_std = stander.fit_transform(Xl)
    imagen = np.float64(Xl_std[:, (range(0, 220))])
    modelostd, predictionstd = kmeans(imagen,'std')
    Xl_stdclase = np.float64(Xl_std[Yl != 0, :])
    modelos1 = KMeans(n_clusters=16, random_state=42).fit(Xl_stdclase)
    modelos['Kmeans 16 std label']=modelos1.labels_ + 1
    Y_final_orig[Yl != 0, 0] = modelos1.labels_ + 1  # Kmeans empieza a etiquetar en 0
    plt.imshow(Y_final_orig.reshape((145, 145), order="F"))
    plt.title('K-means std 16 clusters, etiquetados')
    plt.show()
    # 4º Aprox: PCA
    data=pcapply(Xl_std)
    modelosPCA, predictionsPCA=kmeans(data[Yl!=0,:],'PCA + std ',labeled=True)
    modelos['PCA'] = modelosPCA[2].labels_
    # 5º Aprox: Gaussian Mixtures
    GM = mixture.GaussianMixture(n_components=16, random_state=42).fit_predict(data[Yl!=0,:])
    Y_GM=np.zeros((Yl.shape[0], 1))
    Y_GM[Yl != 0, 0] = GM + 1
    plt.imshow(Y_GM.reshape((145, 145), order="F"))
    plt.title('Gaussian Mixtures, std + PCA 16 clusters, etiquetados')
    plt.show()
    modelos['Gaussian Mixtures']=GM+1
    plotmetrics(Yl,modelos)
    # 6º Clasificación. Selección de Muestras.
    elbow(Xl,Yl)
    Nclusters=[1,4,4,4,3,2,4,3,11,6,1,2,1,4,1,1]
    indexes=muestreo(Xl,Yl,Nclusters)
    # Dibujamos las imagenes
    ax=plt.subplot(1,2,1)
    ax.imshow(X[:,:,1]), ax.axis('off'), plt.title('Image')
    ax=plt.subplot(1,2,2)
    ax.imshow(Y), ax.axis('off'), plt.title('Ground Truth')
