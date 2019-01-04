# Trabajo Machine Learning I

Se plantea una aplicación de clasificación de imágenes multi-espectrales de satélite en la cual se dispone de un conjunto de muestras etiquetadas con sus clases correspondientes.

Una de las características importantes de las imágenes de satélite suele ser el gran volumen de datos que se maneja (aumenta de forma cuadrática con las dimensiones espaciales y lineal con el número de bandas o canales). La otra suele ser el reducido número de muestras etiquetadas de las que se dispone, debido al coste que supone etiquetar las muestras. Por eso mismo vamos a abordar el problema de forma no supervisada empleando todos los datos (clustering) y de forma supervisada pero empleando un subconjunto de datos etiquetados reducido (clasificación). 

En todos los casos, aunque tratemos con imágenes, consideraremos que la información relevante se encuentra en el espectro de cada pixel y no hay que utilizar relaciones espaciales entre pixeles. Es decir, cada pixel es una muestra.

##OBJETIVOS DEL PROYECTO


* Usar algún algoritmo de clustering sobre todos los datos sin emplear las etiquetas para obtener una primera clasificación. Definir una estrategia para clasificar nuevos ejemplos y obtener el mapa de clasificación final de toda la imagen. Obviamente, al no haber empleado las etiquetas de las clases puede que nuestro mapa de clasificación basado en clustering no tenga mucha relación con las clases predefinidas.
* Para aplicar algoritmos de aprendizaje supervisado simularemos la escasez de muestras etiquetadas empleando solo 5000 muestras con sus etiquetas. Sin embargo, la selección de estas muestras no la vamos a hacer de forma aleatoria sino que vamos a emplear algoritmos de clustering para reducir el número de muestras pero preservando la distribución de los datos de entrada y de las clases definidas.  Primero, separar los datos en función de su etiqueta de clase. En cada subconjunto, aplicar algún algoritmo de clustering y definir una estrategia para obtener un subconjunto reducido que sea representativo del conjunto inicial.
* Utilizar algoritmos supervisados sobre el conjunto reducido de entrenamiento para obtener la clasificación de la imagen. Se debe obtener una aproximación del error de clasificación en test utilizando una partición del conjunto de entrenamiento. Se pueden obtener también resultados cualitativos representado las imágenes de clasificación.
* La imagen puede contener bandas con datos erróneos, o con bandas que tengan poca relevancia en el resultado de la clasificación. Utilizando métodos de clasificación que proporcionen un 'ranking' de características, intentar detectar y eliminar aquellas bandas que empeoran (o no mejoran) el resultado de la clasificación. De nuevo, dar un resultado cuantitativo del error en test, y cualitativo mediante la representación de mapas de clasificación.
* Por último, empleando solo el conjunto reducido de muestras etiquetadas y las bandas seleccionadas, utilizar alguna estrategia de combinación de clasificadores (métodos 'ensemble') para intentar mejorar los resultados de la clasificación.

## CONJUNTO DE DATOS

El fichero `datasetB1.mat` contiene los siguientes datos correspondientes a una image multi-espectral:

* `X`:  datos de la imagen (filas x columnas x bandas espectrales)
* `Xl`: datos de la imagen de los cuales se conoce la etiqueta de su clase (muestras x bandas espectrales)
* `Yl`: clases (etiquetas) correspondiente a los datos Yl (muestras x 1)
Existen otros datos en el fichero pero no son de interés para la aplicación a desarrollar.