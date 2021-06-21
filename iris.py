import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
import streamlit as st

st.write("""
# Iris Flower Classifier App
## This web app classifies whether the given iris flower is Versicolor, Vergisinica or Setosa
""")

def user_parameter_input():
    Sepallength=st.sidebar.slider("Sepal length",4.1,1.2,7.9)
    Sepalwidth=st.sidebar.slider("Sepal Width",2.1,2.9,3.8)
    Petallength=st.sidebar.slider("Petal Length",1.0,2.9,7.0)
    Petalwidth=st.sidebar.slider("Petal Width",0.0,1.5,3.0)
    data={
        'sepal length (cm)':Sepallength,
        'sepal width (cm)':Sepalwidth,
        'petal length (cm)':Petallength,
        'petal width (cm)': Petalwidth,
    }
    features=pd.DataFrame(data,index=[0])
    return features
df_features=user_parameter_input()
dataset=load_iris()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['Target']=dataset.target
x=dataset.data
y=dataset.target
from sklearn.ensemble import AdaBoostClassifier
params={
    'n_estimators':[20,50,70,100]
}
from sklearn.model_selection import RandomizedSearchCV
rsc=RandomizedSearchCV(AdaBoostClassifier(),param_distributions=params,
cv=5,return_train_score=False)
rsc.fit(x,y)
pred=rsc.predict(df_features)
st.subheader("The data and their index is:")
st.write(df_features)
st.write("## Predicted class is:")
if pred==0:
    st.write("Setosa")
elif pred==1:
    st.write("Versicolor")
else: 
    st.write('Verginica')
df.to_csv("iris.csv")