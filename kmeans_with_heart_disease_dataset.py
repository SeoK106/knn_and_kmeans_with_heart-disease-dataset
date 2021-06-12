# clustering heart disease dataset with k-means

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
%matplotlib inline

# call and process data
def get_data():
    '''
    import and refine dataset

    - Inputs: raw data
            ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num(target)'] 
    - Outputs: pandas DataFrame
            remove rows which include NaN and change the target data to 0 and 1.

        cf. target data(num) info: diagnosis of heart disease (angiographic disease status)
                                   It is integer valued from 0 (no presence) to 4. 
                                    -> Value 0: < 50% diameter narrowing
                                       Value 1: > 50% diameter narrowing
                                                    (in any major vessel)

    for more infromation -> visit this web page : https://archive.ics.uci.edu/ml/datasets/Heart+Disease
    '''

    # call data file
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
    url ='https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    df = pd.read_csv(url,index_col=False, header=None, names=cols)

    # replace '?' with 'np.nan'
    df = df.replace({'?':np.nan})

    # drop rows with nans for now, only looses 6 rows.
    df.dropna(inplace=True)
    df = df.astype('float64')

    # to change the target data to 0 and 1
    # 0 means 'No heart disease', 1 means 'heart disease'
    df['num'] = df['num']>0
    df['num'] = df['num'].map({False:0, True:1})
    
    return df
    
def draw_clusters(table):
    # run PCA to reduce the dimension 
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(table)
    pca_df = pd.DataFrame(new_data,columns=['principal component1','principal component2'])

    # drawing
    sns.scatterplot(x="principal component1", y="principal component2", hue=table['cluster'], data=pca_df)
    plt.title('K-means Clustering in 2D')
    plt.show()


if __name__=="__main__":
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    
    df = get_data()

    # seperate target(y) from features(x) and scale x
    y = df.pop('num').values
    x = pd.DataFrame(scale(df.values),columns=cols)

    # k-means clusteing
    estimator = KMeans(n_clusters=2, init='k-means++', random_state=42)
    estimator.fit_predict(x)
    
    # create a table containing clustered data 
    table = x.copy()
    table['cluster'] = estimator.labels_
    
    # draw clusters in 2D
    draw_clusters(table)
