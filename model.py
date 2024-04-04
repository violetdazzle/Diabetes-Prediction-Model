import tkinter as tk
from tkinter import messagebox
from Pillow import Image, ImageTk
import pandas as panda 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

def model_create(root):
    # Load data
    df = pd.read_csv("diabetes.csv")
    X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']],df.Outcome,test_size=0.2, random_state=25)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)
    
    # Build and train model
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))   # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=500, verbose=0)

    _, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    messagebox.showinfo("Model Accuracy", f"Model Accuracy: {accuracy*100:.2f}%")

    root.mainloop()
    