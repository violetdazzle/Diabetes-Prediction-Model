import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#from model import model as initialize_model

input_values = []
model = Sequential()
def predict_click(root,predict_button):
    
    # Destroy the predict button
    predict_button.destroy()
    # Create a new frame for input fields
    frame = tk.Frame(root, width=400, height=200)
    frame.place(relx=0.5, rely=0.6, anchor="center")
    
    # List of attributes
    attributes = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']

    
    entry_boxes = []
    for i, attribute in enumerate(attributes):
        # Label
        label = tk.Label(frame, text=attribute)
        label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
        
        # Entry box
        entry_var = tk.StringVar()
        entry_box = tk.Entry(frame, textvariable=entry_var)
        entry_box.grid(row=i, column=1, padx=10, pady=5)
        
        entry_boxes.append(entry_box)

    
    def check_inputs():
        for entry_box in entry_boxes:
            value = entry_box.get()
            if not value.isdigit() and not value.replace('.', '', 1).isdigit():
                messagebox.showerror("Input Error", "Please enter valid integer or float values.")
                return
            input_values.append(float(value))
        
        # Use the model to predict
        #model.predict(sc.transform(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])))
        predicted_label = predict_diabetes()
        print("Pl",predicted_label)
        # Destroy the frame containing the input attributes
        frame.destroy()
        
        # Change background image based on prediction result
        print("Predicted Value is",predicted_label[0])
        if predicted_label[0] <= 0.5:
            messagebox.showinfo("Prediction Result", "You are not diabetic.")
            # Set background image for not diabetic
            background_image = Image.open("images\\dia.png")
        else:
            messagebox.showinfo("Prediction Result", "You have diabetes.")
            # Set background image for diabetic
            background_image = Image.open("images\\notdia.png")
    
    # Resize the background image to fit the window
        new_width = int(background_image.width * 0.7)
        new_height = int(background_image.height * 0.7)
        background_image = background_image.resize((new_width, new_height))
        root.background_image_tk = ImageTk.PhotoImage(background_image)
        background_image.configure(image=root.background_image_tk)
    submit_btn = tk.Button(frame, text="Submit", command=check_inputs)
    submit_btn.place(relx=0.5, rely=0.9, anchor="center")
    
    # Predict button for input submission



def model_create(root):
    # Load data
    print("model reached")
    df = pd.read_csv("diabetes.csv")
    X_train, X_test, y_train, y_test = train_test_split(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction', 'Age']],df.Outcome,test_size=0.2, random_state=25)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)
    print("done preprocessing")
    # Build and train model
    model.add(Dense(16, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))   # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=500, verbose=0)
    print("accuracy calculation")
    accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(accuracy)
    messagebox.showinfo("Model Accuracy", f"Model Accuracy: {accuracy[1]*100:.2f}%")


def predict_diabetes():
    print(input_values)
    # model.predict(sc.transform(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])))
    #StandardScaler().fit_transform(X_train)
    prediction =model.predict(StandardScaler().transform(np.array([input_values])))
    print(prediction[0])
    return prediction

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Prediction of Diabetes")

    # Load background image
    background_image = Image.open("images/background.png")
    new_width = int(background_image.width * 0.7)
    new_height = int(background_image.height * 0.7)
    background_image = background_image.resize((new_width, new_height))
    root.background_image_tk = ImageTk.PhotoImage(background_image)  # Store image object as attribute of root

    # Create background label
    background_label = tk.Label(root, image=root.background_image_tk)
    background_label.place(relx=0.5, rely=0.5, anchor="center")
    root.geometry(f"{new_width}x{new_height}")

    # Predict button
    predict_button_image = Image.open("images/predict button.png")
    button_width = int(predict_button_image.width * 0.7)
    button_height = int(predict_button_image.height * 0.7)
    predict_button_image = predict_button_image.resize((button_width, button_height))
    predict_button_image_tk = ImageTk.PhotoImage(predict_button_image)
    predict_button = tk.Button(root, image=predict_button_image_tk, 
                               command=lambda: predict_click(root,predict_button),
                               borderwidth=0, highlightthickness=0)
    predict_button.place(relx=0.5, rely=0.9, anchor="center")

    root.mainloop()