# Trabajo Machine Learning
# @authors: Vicente Cifre, Mikel Ardanaz
import numpy as np
import re
import scipy.io.matlab as matlab
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import v_measure_score, adjusted_rand_score, mutual_info_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.cluster import KElbowVisualizer


def kmeans(imagen, tipo='', labeled=False):
    """
    Realiza un clustering utilizando el algoritmo KMeans implementado en scikit-learn con 5,10 y 17 clusters.
    Posteriormente realiza una gráfica con los resultados.
    :return: * modelos, predictions: El modelo ya entrenado y el resultado de la prediccion
    """
    modelos = []
    predictions = []
    if labeled:
        clusters = [5, 10, 16]
    else:
        clusters = [5, 10, 17]
    for i, nclusters in enumerate(clusters):  # Yl va de 0 a 16
        kmeans = KMeans(n_clusters=nclusters, random_state=42,
                        n_jobs=-1).fit(imagen)  # njobs=-1-> Parallel processing
        modelos.append(kmeans)
        # Prescindible
        # (https://stackoverflow.com/questions/25012342/scikit-learns-k-means-what-does-the-predict-method-really-do)
        predictions.append(kmeans.predict(imagen))
        ax = plt.subplot(1, 3, i + 1)
        if labeled:
            plt.suptitle('Kmeans etiquetado')
            Y_final_orig[Yl != 0, 0] = modelos[i].labels_ + 1
            predictions[i] = Y_final_orig
        else:
            plt.suptitle('Kmeans')
        ax.imshow(predictions[i].reshape((145, 145), order="F"))
        plt.title(tipo + str(modelos[i].n_clusters) + ' clusters')  # Dims(Y)
    plt.show()
    return modelos, predictions


def pcapply(X):
    """
    Realiza un PCA y muestra en una gráfica la explicación de la variabilidad respecto al nº de vars. La hemos usado
    para elegir el número de variables que queriamos conservar.
    :param X:Dataset
    :return: data -  Conjunto de datos reducido
    """
    data = PCA(n_components=40).fit_transform(X)
    _, b, _ = np.linalg.svd(X.transpose().dot(X))  # Demo mejor con 40
    plt.title('Perdida Explicación variabilidad en base al número de variables')
    plt.plot(range(10, 75), b[10:75], 'bx-')
    plt.show()
    return data


def plotmetrics(Yl, modelos):
    """
    Muestra en una gráfica algunas de los medidas de bondad estudiadas
    :param Yl: Etiquetas de clases
    :param modelos: Diccionario con NombreModelo y clases asignadas
    :return:
    """
    mutualinfo = {}
    vmeasure = {}
    rand = {}
    for i in modelos:
        mutualinfo[i] = mutual_info_score(Yl[Yl != 0], modelos[i])
        vmeasure[i] = v_measure_score(Yl[Yl != 0], modelos[i])
        rand[i] = adjusted_rand_score(Yl[Yl != 0], modelos[i])
    mutualinfo['Ground Truth'] = mutual_info_score(Yl[Yl != 0], Yl[Yl != 0])
    vmeasure['Ground Truth'] = v_measure_score(Yl[Yl != 0], Yl[Yl != 0])
    rand['Ground Truth'] = adjusted_rand_score(Yl[Yl != 0], Yl[Yl != 0])
    plt.subplot(221)
    plt.bar(range(len(mutualinfo)), mutualinfo.values(),
            align='center')  # (Nombres en mutualinfo.keys)
    plt.title('Información Mutua')
    plt.subplot(222)
    plt.bar(range(len(vmeasure)), vmeasure.values(), align='center')
    plt.title('V Measure')
    plt.subplot(223)
    plt.bar(range(len(rand)), rand.values(), align='center')
    plt.title('Rand')
    plt.show()


def elbow(Xl, Yl):
    """
    Implementación del método Elbow (https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set)
    para conocer el número de clusters que debe tener cada dataset.Lo usaremos posteriormente en la selección de muestras.
    El número de clusters se elige a ojo en función de donde encontremos el codo y del tiempo necesario para el aprendizaje.
    IMP: En algunos casos no esta bien definido el codo. (¡La vida real es asi de dura!)
    """
    for clase in range(1, 17):
        indexclasified = np.where(Yl == clase)[0]  # Indexes of class
        Xlclase = Xl[indexclasified, :]
        model = KMeans(n_jobs=-1)
        visualizer = KElbowVisualizer(model, k=(1, 12), title=(
            'Método Elbow para la clase ') + str(clase))
        visualizer.fit(Xlclase)
        visualizer.poof()


def seleccionPuntos(clasificacion, total=5000):
    """
    Implementación proporcional respecto a la aparición de cada clase. Rara vez tendremos 5000, lo habitual es tener alguno menos.
    :param clasificacion: Criterio usado para para la selección de elementos
    :param total: Número de elementos a particionar
    :return: npuntos -- puntos por clase/cluster
    """
    npuntos = []
    for clase in np.unique(clasificacion):
        # Resultados truncados (No podemos tener medio dato)
        npuntos.append(
            int(total * np.sum(clasificacion == clase) / clasificacion.shape[0]))
    return npuntos


def muestreo(Xl, Yl, Nclusters):
    """
    Implementa la mixtura de gaussianos ya que es el clustering que mejores resultados nos ha dado.
    Selecciona ptos. + cerca de las medias. Podría no generalizar bien ya que no sería representativo,
    pero después de tener que hacer un clustering no voy a elegir los puntos de forma aleatoria.
    :param Xl: Muestras etiquetadas
    :param Yl: Etiquetas de clases
    :param Nclusters: lista con nº clusters por clase (Obtenidos vía elbow)
    :return: Índices de los puntos representantes
    """
    ptsxclase = seleccionPuntos(Yl[Yl > 0])  # Puntos para cada clase
    Yl_final = np.zeros(Yl.shape[0])
    for clase in np.unique(Yl[Yl > 0]):
        indexclasified = np.where(Yl == clase)[0]
        GM = GaussianMixture(n_components=Nclusters[
            clase - 1]).fit(Xl[indexclasified, :])
        predictions = GM.predict(Xl[indexclasified, :])
        ptsxcluster = seleccionPuntos(predictions, total=ptsxclase[clase - 1])
        Yl_Cluster = np.zeros(Yl.shape[0])
        for i, centro in enumerate(GM.means_):
            # Index of samples belonging to cluster
            indexofcluster = indexclasified[predictions == i]
            Xl_Cluster = Xl[indexofcluster, :]  # Cogemos Xl
            # Diferencia punto con centro
            norma = np.sqrt(np.sum((Xl_Cluster - centro) ** 2, axis=1))
            # Nos quedamos con ptsxcluster elementos
            nearestindexes = norma.argsort()[:ptsxcluster[i]]
            Yl_Cluster[indexofcluster[nearestindexes]] = i + 1
        Yl_final[Yl_Cluster > 0] = clase
    return np.where(Yl_final > 0)[0]


def clasifica(Clasificador, name, X_train, Y_train, X_test, Y_test, index_test):
    """
    Esta función realiza el proceso de fit y predict habitual
    :param index_test:
    :param name:
    :param Clasificador: Clasificador a entrenar
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :return:
    """
    Clasificador.fit(X_train, Y_train)
    pred = Clasificador.predict(X_test)
    Yl_prediccion = np.zeros(Yl.shape[0])
    Yl_prediccion[index_test] = pred
    plt.imshow(np.reshape(Yl_prediccion, (145, 145), order="F")),
    plt.axis('off'),
    plt.title(re.compile('.*\(').findall(str(Clasificador))[0][:-1])
    plt.show()
    print('Clasificador: ', name)
    print('Aciertos:', np.sum((Y_test - pred) == 0))
    print('Fallos:', np.sum((Y_test - pred) != 0))
    precisionTest = np.mean((Y_test - pred) == 0)
    print('Proporción de aciertos (precisión en Test):', precisionTest)
    return Clasificador, precisionTest


def PredictOthers(Clasificador, Xl, Yl, otherindexes):
    """
    Realiza la predicción con los indices no seleccionados y los muestra gráficamente
    :param Yl:
    :param Clasificador:
    :param Xl:
    :param otherindexes:
    :return:
    """
    pred = Clasificador.predict(Xl[otherindexes, :])
    Yl_prediccion = np.zeros(Yl.shape[0])
    Yl_prediccion[otherindexes] = pred
    plt.imshow(np.reshape(Yl_prediccion, (145, 145), order="F")),
    plt.axis('off'),
    # re.compile('.*\(').findall(str(Clasificador))[0][:-1] obtiene el nombre
    # del modelo
    plt.title(re.compile('.*\(').findall(str(Clasificador))[0][:-1])
    plt.show()
    precisionOtros = np.mean((Yl[otherindexes] - pred) == 0)
    print('Proporción de aciertos (precisión en Otros) :', precisionOtros)
    return precisionOtros


def ranking(Xl_reduced, Yl_reduced):
    """
    https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    :param Xl_reduced:
    :param Yl_reduced:
    :return:
    """
    parameters = {'criterion': ['gini', 'entropy'],
                  'max_depth': [4, 10, 20],
                  'min_samples_split': [2, 4, 8],
                  }
    clf = GridSearchCV(ExtraTreesClassifier(class_weight='balanced', n_estimators=10), parameters, verbose=3,
                       cv=5, n_jobs=-1)
    clf.fit(Xl_reduced, Yl_reduced)
    clf = clf.best_estimator_
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    print(importances[indices])
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(Xl_reduced.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(Xl_reduced.shape[1]), indices)
    plt.xlim([-1, Xl_reduced.shape[1]])
    plt.show()
    imp_threshold = 0.0085
    idx_final = np.where(importances > imp_threshold)[0]
    final_X_train = X_train[:, idx_final]
    final_X_test = X_test[:, idx_final]
    return final_X_train, final_X_test


def ensemble(final_X_test, final_X_train, Y_train, Yl, index_train):
    """
    Realiza un ensemble de 2 niveles utilizando los clasificadores que mejores resultados nos han dado.
    Posteriormente plotea los resultados.
    :param final_X_test:
    :param final_X_train:
    :param Y_train:
    :param Yl:
    :param index_train:
    :return:
    """
    ensemble_models = [DecisionTreeClassifier(),
                       LinearSVC(),
                       GaussianNB(),
                       LogisticRegression(solver='lbfgs', multi_class='auto'),
                       SVC(kernel="linear", C=0.025)]
    n_folds = len(ensemble_models)
    kf = KFold(n_folds, shuffle=True)
    X_lv2 = np.zeros((final_X_train.shape[0], n_folds))
    y_lv2 = np.zeros(Y_train.shape)
    for itrain, itest in kf.split(final_X_train):
        y_lv2[itest] = Y_train[itest]
        # Train
        for n in range(n_folds):
            ensemble_models[n].fit(final_X_train[itrain, :], Y_train[itrain])
            X_lv2[itest, n] = ensemble_models[
                n].predict(final_X_train[itest, :])
    # Nivel 2
    Clas_lv2_m2 = SVC(kernel="linear")
    Clas_lv2_m2.fit(X_lv2, y_lv2)
    # Train
    for n in range(n_folds):
        ensemble_models[n].fit(final_X_train, Y_train)
    # Predicción
    Ypred_test = np.zeros((Y_test.shape[0], n_folds))
    Ypred_excl = np.zeros((final_X_train.shape[0], n_folds))
    for n in range(n_folds):
        Ypred_test[:, n] = ensemble_models[n].predict(final_X_test)
        Ypred_excl[:, n] = ensemble_models[n].predict(final_X_train)
    yc2 = Clas_lv2_m2.predict(Ypred_excl)
    Yl_prediccion = np.zeros(Yl.shape[0])
    Yl_prediccion[index_train] = yc2
    plt.imshow(np.reshape(Yl_prediccion, (145, 145), order="F")),
    plt.axis('off'),
    plt.title('Ensemble')
    plt.show()


if __name__ == '__main__':
    # Al importar yellowbrick, se cambia el esquema de colores a grayscale.
    plt.style.use('default')
    # Lectura de la imagen de fichero de Matlab .mat
    mat_file = "datasetB1.mat"
    mat = matlab.loadmat(mat_file, squeeze_me=True)  # devuelve un dictionary
    list(mat.keys())  # variables almacenadas
    # Lectura de los datos
    X = mat["X"]  # imagen (hipercubo 3D: filas x columnas x variables)
    Xl = mat["Xl"]  # muestras etiquetadas (muestas x variables)
    Yl = mat["Yl"]  # etiquetas de clases (muestras x 1, 0=sin clase)
    del mat
    modelos = {}  # Dict para almacenar los modelos etiquetados

    # Reshape del Ground Truth como una imagen
    Y = np.reshape(Yl, (X.shape[0], X.shape[1]), order="F")
    # imagen en sus 220 dimensiones
    imagen = np.float64(Xl[:, (range(0, 220))])
    # 1º Aproximación: Kmeans 5,10,17 clusters
    modelo, predictions = kmeans(imagen)
    # 2º Aprox: Filter background
    # Probamos ahora a hacer cluster solo de los que tienen etiqueta (al resto
    # les mantendremos el valor de 0)
    Y_final_orig = np.zeros((Yl.shape[0], 1))
    Xl_SoloClasei = np.float64(Xl[Yl != 0, :])
    modelos1 = KMeans(n_clusters=16, random_state=42,
                      n_jobs=-1).fit(Xl_SoloClasei)
    modelos['Kmeans 16 label'] = modelos1.labels_ + 1
    Y_final_orig[Yl != 0, 0] = modelos1.labels_ + \
        1  # Kmeans empieza a etiquetar en 0
    plt.imshow(Y_final_orig.reshape((145, 145), order="F"))
    plt.title('K-means 16 clusters, solo etiquetados')
    plt.show()
    # 3º Aprox: Repetimos 1º y 2º Aprox. con estandarización
    stander = StandardScaler()
    Xl_std = stander.fit_transform(Xl)
    imagen = np.float64(Xl_std[:, (range(0, 220))])
    modelostd, predictionstd = kmeans(imagen, 'std')
    Xl_stdclase = np.float64(Xl_std[Yl != 0, :])
    modelos1 = KMeans(n_clusters=16, random_state=42,
                      n_jobs=-1).fit(Xl_stdclase)
    modelos['Kmeans 16 std label'] = modelos1.labels_ + 1
    Y_final_orig[Yl != 0, 0] = modelos1.labels_ + \
        1  # Kmeans empieza a etiquetar en 0
    plt.imshow(Y_final_orig.reshape((145, 145), order="F"))
    plt.title('K-means std 16 clusters, etiquetados')
    plt.show()
    # 4º Aprox: PCA
    data = pcapply(Xl_std)
    modelosPCA, predictionsPCA = kmeans(
        data[Yl != 0, :], 'PCA + std ', labeled=True)
    modelos['PCA'] = modelosPCA[2].labels_
    # 5º Aprox: Gaussian Mixtures
    GM = GaussianMixture(
        n_components=16, random_state=42).fit_predict(data[Yl != 0, :])
    Y_GM = np.zeros((Yl.shape[0], 1))
    Y_GM[Yl != 0, 0] = GM + 1
    plt.imshow(Y_GM.reshape((145, 145), order="F"))
    plt.title('Gaussian Mixtures, std + PCA 16 clusters, etiquetados')
    plt.show()
    modelos['Gaussian Mixtures'] = GM + 1
    plotmetrics(Yl, modelos)
    # 6º Clasificación. Selección de Muestras.
    elbow(Xl, Yl)
    Nclusters = [1, 4, 4, 4, 3, 2, 4, 3, 11, 6, 1, 2, 1, 4, 1, 1]
    indexes = muestreo(Xl, Yl, Nclusters)
    Yl_reduced = Yl[indexes]
    Xl_reduced = Xl[indexes, :]
    otherindexes = np.setdiff1d(np.where(Yl > 0), indexes)
    # 7º Clasificación. Partición en 2 subconjuntos: Train y Test
    X_train, X_test, Y_train, Y_test, index_train, index_test = train_test_split(Xl_reduced, Yl_reduced,
                                                                                 indexes,
                                                                                 test_size=.33, random_state=42)
    # 8º Clasificación. Llamada a los clasificadores
    testError = {}
    otherError = {}
    # PARAMETERS
    # Habíamos probado con entre 2 y 75 pero apenas hay mejoría. Además al
    # considerar más vecinos. Aumenta el overfitting y el tiempo.
    neighbors = 2
    depth = 10
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # LinearSVC is Similar to SVC with parameter kernel=’linear’, but
    # implemented in terms of liblinear rather than libsvm, so it has more
    # flexibility in the choice of penalties and loss functions and should
    # scale better to large numbers of samples. ->
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    names = ['KNeighborsClassifier', 'DecisionTreeClassifier', 'NearestCentroid', 'LogisticRegression',
             'RandomForestClassifier', 'LinearSVC', 'SVC', 'GaussianNB', 'BernoulliNB', 'AdaBoostClassifier']
    classifiers = [KNeighborsClassifier(
        n_neighbors=neighbors, n_jobs=-1), DecisionTreeClassifier(max_depth=depth), NearestCentroid(),
        LogisticRegression(class_weight='balanced', n_jobs=-1),
        RandomForestClassifier(
            max_depth=12, n_estimators=10, n_jobs=-1), LinearSVC(),
        SVC(kernel="linear", C=0.025),
        GaussianNB(), BernoulliNB(),
        AdaBoostClassifier(n_estimators=100)]
    print('Total Puntos:', Y_test.shape[0])
    for name, clf in zip(names, classifiers):
        clf_entrenado, precisionTest = clasifica(
            clf, name, X_train, Y_train, X_test, Y_test, index_test)
        precisionOtros = PredictOthers(clf_entrenado, Xl, Yl, otherindexes)
        testError[name] = precisionTest
        otherError[name] = precisionOtros
    # 9º Ranking de características;
    final_X_train, final_X_test = ranking(Xl_reduced, Yl_reduced)
    # 10º Ensembles
    # https://scikit-learn.org/stable/modules/ensemble.html
    ensemble(final_X_test, final_X_train, Y_train, Yl, index_train)
    # Accuracy
    plt.subplot(221)
    plt.bar(range(len(testError)), testError.values(),
            align='center')
    plt.title('Precisión Test')
    plt.subplot(222)
    plt.bar(range(len(otherError)), otherError.values(), align='center')
    plt.title('Precisión otros')
    plt.show()
    # Dibujamos las imagenes
    ax = plt.subplot(1, 2, 1)
    ax.imshow(X[:, :, 1]), ax.axis('off'), plt.title('Image')
    ax = plt.subplot(1, 2, 2)
    ax.imshow(Y), ax.axis('off'), plt.title('Ground Truth')
    plt.show()
