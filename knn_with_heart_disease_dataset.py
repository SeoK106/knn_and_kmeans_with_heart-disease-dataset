# practice Knn with heart disease dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.preprocessing import scale

# call and refine data
def get_data():
    '''
    Data information

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

def calc_performance(y_true,y_pred):
    
    print("** Performance of the knn model **\n")
    
    # accuracy = # of correctly predicted target/total number of dataset = (tp+tn)/(tp+fp+tn+fn)
    acc = accuracy_score(y_true,y_pred)
    print(f"  * Accuracy score: {acc:.2f}")

    '''confusion_matrix
     In the binary case, we can extract true positive, false positive, true negative, false negative.
     Shape of the confusion matrix from sklearn metrics as follow:
        [[ tn, fp ],
         [ fn, tp ]]
    '''
    cm = confusion_matrix(y_true,y_pred)
    tn,fp,fn,tp = cm.ravel()
   
    # Plotting confision matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot() 
    

    # precision = tp/(tp+fp): High precision means performance is good -> best is 1, worst is 0.
    pre = precision_score(y_true,y_pred)
    print(f"  * precision: {pre:.2f}")

    # recall(sensitivity) = tp/(tp+fn): The best is 1 and the worst is 0.
    rec = recall_score(y_true,y_pred)
    print(f"  * recall: {rec:.2f}")
    
    # specificity (true negative rate) = tn/(tn+fp)
    tnr = tn/(tn+fp)
    print(f"  * true negative rate: {tnr:.2f}")
    

    # f-measure = 2*precision*recall / (precision_recall) -> the harmonic mean of precision and recall
    fs = f1_score(y_true,y_pred)
    print(f"  * f1-score: {fs:.2f}")
    
    # This function will show you {precision,recall,f1-score}
    #from sklearn.metrics import classification_report
    #tmp = classification_report(y_true,y_pred)


if __name__=="__main__":
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    
    df = get_data()

    # seperate target(y) from features(x) and scale x
    y = df.pop('num').values
    x = pd.DataFrame(scale(df.values),columns=cols)

    # KNN

    # ~ 1. Applying K-Fold
    print("\n>> knn with k-fold \n")
    
    kf = KFold(n_splits = 3, shuffle = True, random_state = 106)
    knnc= KNeighborsClassifier(n_neighbors = 5)

    seq = 1
    acc_list = []
    for train_idx, test_idx in kf.split(x):
        x_train = x.iloc[train_idx]
        y_train = y[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y[test_idx]

        # fitting model with folded train data
        model = knnc.fit(x_train,y_train)

        # predict with folded testset
        y_pred = model.predict(x_test)
        
        acc = accuracy_score(y_test,y_pred) 
        acc_list.append(acc)
        print(f"{seq}: # of Train set: {x_train.shape[0]}, # of validation set: {x_test.shape[0]}, Accuracy: {acc:.2f}")   
        seq += 1
    
    print(f"\n  * Average accuracy: {np.mean(acc_list):.2f}")    
    
    '''
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, x_train, y_train, cv=3)
    print(scores)
    print(f"Average accuracy: {scores.mean()}")
    '''
          
    # ~1.1 Advanced K-Fold with StratifiedKFold
    print("\n>> knn with Stratified k-fold \n")
    
    skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 106)
    knnc= KNeighborsClassifier(n_neighbors = 5)

    seq = 1
    acc_list = []
    for train_idx, test_idx in skf.split(x,y):
        x_train = x.iloc[train_idx]
        y_train = y[train_idx]
        x_test = x.iloc[test_idx]
        y_test = y[test_idx]

        # fitting model with folded train data
        model = knnc.fit(x_train,y_train)

        # predict with folded testset
        y_pred = model.predict(x_test)
        
        acc = accuracy_score(y_test,y_pred) 
        acc_list.append(acc)
        print(f"{seq}: # of Train set: {x_train.shape[0]}, # of validation set: {x_test.shape[0]}, Accuracy: {acc:.2f}")    
        seq += 1
    
    print(f"\n  * Average accuracy: {np.mean(acc_list):.2f}")
    

    # ~ 2. without k-fold

    # split data into train and test data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=106)

    # fitting model with train data
    knnc = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    model = knnc.fit(x_train,y_train)

    # predict with test data
    y_pred = model.predict(x_test)

    print("-"*100)
    print("\n>> knn without k-fold \n")
    calc_performance(y_test,y_pred)
