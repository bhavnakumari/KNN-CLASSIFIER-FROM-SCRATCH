import pandas as pd
import numpy as np
from KNearestNeighbor import KNearestNeighbors

data=pd.read_csv('Social_Network_Ads.csv')

X=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

#print(X.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#An object of KNN
knn=KNearestNeighbors(k=5)

knn.fit(X_train,y_train)

knn.predict(np.array([60,100000]).reshape(1,2))

def predict_new():
    age=int(input("Enter the age:"))
    salary=int(input("Enter the salary:"))

    X_new=np.array([[age],[salary]]).reshape(1,2)
    X_new=scaler.transform(X_new)
    result=knn.predict(X_new)

    if result==0:
        print("Will not Purchase")
    else:
        print("Will Purchase")

predict_new()