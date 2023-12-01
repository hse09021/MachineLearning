import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from xgboost import XGBClassifier
import time

#get data
training_data = pd.read_csv('./fashionmnist/fashion-mnist_train.csv')

X_test = training_data['label']
X_train = training_data.drop('label',axis=1)

test_data = pd.read_csv('./fashionmnist/fashion-mnist_test.csv')

y_test = test_data.drop('label',axis=1)

#get private data
df_private = pd.read_csv('Private_data.csv')

#normalization
X_train = X_train.astype('float32')
y_test = y_test.astype('float32')
df_private = df_private.astype('float32')

X_train /= 255.0
y_test /=255.0
df_private /=255.0

df_private.columns = ['pixel' + str(i + 1) for i in range(784)]

#split data
seed = 99
np.random.seed(seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                  X_test, 
                                                  test_size=0.1, 
                                                  random_state = seed)

#pca
pca = PCA(n_components=100, random_state=42)
X_train_pca =pca.fit_transform(X_train)
X_test_pca = pca.transform(X_val)
y_test_pca =pca.transform(y_test)
pca_private = pca.transform(df_private)

X_train_PCA1 = pd.DataFrame(X_train_pca)
X_test_PCA1 = pd.DataFrame(X_test_pca)
pca_private_1 = pd.DataFrame(pca_private)

#svc
svc = SVC(C=13,kernel='rbf',gamma="auto",probability = True)
svc.fit(X_train_PCA1, y_train)

#random forest
random_forest = RandomForestClassifier(criterion='entropy', max_depth=70, n_estimators=100)
random_forest.fit(X_train_PCA1, y_train)

#ensemble
smv_random_forest_ensemble_model = VotingClassifier(estimators=[('svm', svc), ('random_forest', random_forest)], voting='soft')
smv_random_forest_ensemble_model.fit(X_train_PCA1,y_train)

#inference
private_pred = smv_random_forest_ensemble_model.predict(pca_private_1)

#create list
private_max_digits = len(str(len(private_pred) - 1)) 
private_index = [f"{i:0{private_max_digits}}" for i in range(len(private_pred))]

# 두 개의 리스트
list1 = private_index
list2 = private_pred

# 두 개의 리스트 합성
combined_lists = list(zip(list1, list2))

# NumPy 배열로 변환하여 2차원 배열 생성
array_2d = np.array(combined_lists)

np.savetxt("testResult.txt", combined_lists, fmt='%s', delimiter=" ")
