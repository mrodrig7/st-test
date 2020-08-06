# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 09:45:44 2020

@author: root
https://www.youtube.com/watch?v=y3qZFu3gmYE
"""
# Description: This is program detects something

# Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

path='.'
img_file=path + '\\foto_01.jpg'
data_file=path + '\\diabetes.csv'

# Create a title and subtitle
st.write("""
# Detection
using machine learning in python!
""")

#Open and display an image
image = Image.open(img_file)
st.image(image, caption='ML', use_column_width=True)

# Get the data
df = pd.read_csv(data_file)

# Set a subheader
st.subheader('Data information: ')
st.dataframe(df)
st.subheader('Data statistics: ')
st.write(df.describe())
st.subheader('Bar chart: ')
chart=st.bar_chart(df)

# Split the data into independent and dependent
X = df.iloc[:,0:8].values
Y = df.iloc[:,-1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)

# Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('glucose',0,199,117)
    blood_pressure = st.sidebar.slider('blood_pressure',0,122,72)
    skin_thickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('insulin',0.0, 846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF=st.sidebar.slider('DPF',0.078,2.42,0.3725)
    age=st.sidebar.slider('age',21,81,29)
    
    #Store a disctionary into a variable
    user_data={'pregnancies': pregnancies,
               'glucose': glucose,
               'blood_pressure': blood_pressure,
               'skin_thickness': skin_thickness,
               'insulin': insulin,
               'BMI': BMI,
               'DPF': DPF,
               'age': age
            }
    
    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

# Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#Set a subheader and display the classsification
st.subheader('Classification: ')
st.write(prediction)