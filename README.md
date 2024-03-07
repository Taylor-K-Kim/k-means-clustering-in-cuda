# k-means-clustering-in-cuda
K-Means Clustering in CUDA with multidimensional datasets.

Description
-----------

There are two datasets used, obtained from UCI Machine Learning Repository.

1. Iris dataset from https://archive.ics.uci.edu/dataset/53/iris
   
   contains N=150 datapoints, F=4 features, K=3 expected clusters
   
3. Wine Quality dataset from https://archive.ics.uci.edu/dataset/186/wine+quality
   
   contains N=1599 datapoints, F=11 features, K=6 expected clusters

For loading these datasets for clustering, last column for target (categorical) values has been removed. However, it is used at the end of the program to check for the clustering correctness, i.e., whether the program outputs the expected clustering.

Notes:
1. Current implementation does not check for convergence in the k-means clustering algorithm rather uses MAX_ITER to repeat the two steps of the algorithm, the assignment and the update part.
2. Expected K-cluster values are assumed by looking at the dataset's target (categorical/classfication) values. Therefore, fine-tuning for choosing the K-values has not been implemented.

