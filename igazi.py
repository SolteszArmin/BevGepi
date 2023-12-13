import streamlit as st
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("Stars.csv")

columNames=['temperature', 'luminosity', 
              'radius', 'absolute_magnitude', 
              'star_color', 'spectral_class', 'star_type']
df.columns = ['temperature', 'luminosity', 
              'radius', 'absolute_magnitude', 
              'star_color', 'spectral_class', 'star_type']

df.replace(['Blue-white','Blue White','Blue white'],'Blue-White',inplace=True)
df.replace(['white','Whitish'], 'White', inplace= True)
df.replace(['yellowish','Pale yellow orange','Orange-Red'], 'Yellowish', inplace= True)
df.replace(['yellow-white','Yellowish White','White-Yellow'], 'Yellow-White', inplace= True)

X=df.drop(['star_type'],axis=1)
y=df['star_type']

oneHotX=pd.get_dummies(X)
oneHotX.head()


X_train,X_test,y_train,y_test=train_test_split(oneHotX,y,test_size=0.2,random_state=0)
logModel=LogisticRegression(max_iter=500,solver='lbfgs')
logModel.fit(X_train,y_train)
logModel.score(X_test,y_test)
st.title("Star Type Predicter")

starTypes={0:'Brown Dwarf',
           1:'Red Dwarf',
           2:'White Dwarf',
           3:'Main Sequence',
           4:'Supergiant',
           5:'Hypergiant'}

with st.form(key='form1'):
        temperature=st.number_input("Temperature:")
        luminosity=st.number_input("Luminosity:")
        radius=st.number_input("Radius:")
        absolute_magnitude=st.number_input("Absolute Magnitude")
        star_color=st.radio(
                "Star Color:",
                ["Red","Blue","Blue-White","Yellow-White","White","Yellowish","Orange"],0,horizontal=True
        )
        orbitalClass=st.radio(
                "Spectral Class:",
                ['O','B','A','F','G','K','M'],0,horizontal=True
        )
        submit=st.form_submit_button(label="Predict")

if submit:
    d={'temperature':[temperature],
        'luminosity':[luminosity],
        'radius':[radius],
        'absolute_magnitude':[absolute_magnitude],
        'star_color_Blue':[0],
        'star_color_Blue-White':[0],
        'star_color_Orange':[0],
        'star_color_Red':[0],
        'star_color_White':[0],
        'star_color_Yellow-White':[0],
        'star_color_Yellowish':[0],
        'spectral_class_A':[0],
        'spectral_class_B':[0],
        'spectral_class_F':[0],
        'spectral_class_G':[0],
        'spectral_class_K':[0],
        'spectral_class_M':[0],
        'spectral_class_O':[0]}

    df=pd.DataFrame(d)
    df['star_color_'+star_color]=
    df['spectral_class_'+orbitalClass]=1

    picturename=""
    prediction=logModel.predict(df)
    predicted=prediction[0]
    st.write(starTypes[predicted])
    if predicted==0:
        picturename="BrownDwarf.jpg"
    elif predicted==1:
        picturename="RedDwarf.png"
    elif predicted==2:
        picturename="WhiteDwarf.jpg"
    elif predicted==3:
        picturename="mainSequence.jpg"
    elif predicted==4:
        picturename="mainSequence.jpg"
    else:
        picturename="hypergiant.jpg"
    
    st.image(picturename, caption=starTypes[predicted])

                
            



