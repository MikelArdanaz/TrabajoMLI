# Documentación para Memoria 
[Link de scikit en el que explican los tipos de clustering](https://scikit-learn.org/stable/modules/clustering.html)
* Muy detallado y muy gráfico. Ideal para hacer copypaste en la memoria.  
* KMeans can be seen as a special case of Gaussian mixture model with equal covariance per component.
* Mencionar que aplicando un `PCA`, Kmeans va más rápido, los resultados de la aplicación y el porque.  
* `AffinityPropagation` y `Mean-Shift` **MUY** lentos. Inviables (Dataset muy grande)  
* `DBSCAN` considera todo de la misma clase  
* `Birch` Más rápido que `Kmeans` pero resultado mucho peor. 

[Explicación Pros & Cons jerarquico VS Kmeans](https://www.quora.com/What-are-the-pros-and-cons-of-k-means-vs-hierarchical-clustering)