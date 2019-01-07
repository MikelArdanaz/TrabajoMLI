# Documentación para Memoria 
[Link de scikit en el que explican los tipos de clustering](https://scikit-learn.org/stable/modules/clustering.html)
* Muy detallado y muy gráfico. Ideal para hacer copypaste en la memoria.  
* KMeans can be seen as a special case of Gaussian mixture model with equal covariance per component.
* Mencionar que aplicando un `PCA`, Kmeans va más rápido, los resultados de la aplicación y el porque.  
* `AffinityPropagation` y `Mean-Shift` **MUY** lentos. Inviables (Dataset muy grande) 
* `Spectral Clustering` también inviable; Graph is not fully connected, spectral embedding may not work as expected.   
* `DBSCAN` considera todo de la misma clase  
* `Birch` Más rápido que `Kmeans` pero resultado mucho peor. 
* `Mixturas Gaussianas` ver Episodio IV Paquito  

[Explicación Pros & Cons jerarquico VS Kmeans](https://www.quora.com/What-are-the-pros-and-cons-of-k-means-vs-hierarchical-clustering)  
[Determining the number of clusters in a data set](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set)  
