Practice knn and kmeans with heart-disease-dataset 
=============

This is the code to practice **knn** and **k-means** with heart disease dataset.<br>
I used sklearn modules like [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and etc.

About Dataset
-----
> **Heart Disease Data**

* Features(13): [ 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
* Target: diagnosis of heart disease (angiographic disease status)
    *  integer valued from 0 (no presence) to 4 -> **binarization**: 0 means 'No heart disease', 1 means 'heart disease'
* For more information, visit a [webpage](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

About Result
-----

### 1. knn 
![knn_result](https://user-images.githubusercontent.com/66738234/121795556-e9fa2e00-cc4c-11eb-8818-5211cfb18049.png)

You can practice the usage and compare [K-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html), [stratified k-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), and normal knn method.<br>
The result is evaluating the preformance of the models. 

### 2. kmeans
![kmeans_result](https://user-images.githubusercontent.com/66738234/121795377-0d23de00-cc4b-11eb-8284-f33c7f0c5088.png)

To draw clusters in 2D, I used [sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce dimension.<br>
That is why the x and y axes are the principal component.

Environment
----
- python 3.7
- Used Modules - **numpy, pandas, matplotlib, seaborn, scikit-learn**
