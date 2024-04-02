import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from input import predict_click
from model import model_create as mc

root = tk.Tk()
root.title("Prediction of Diabetes")


# Background image
background_image = Image.open("images/background.png")
new_width = int(background_image.width * 0.7)  
new_height = int(background_image.height * 0.7) 
background_image = background_image.resize((new_width, new_height))
background_image_tk = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_image_tk)
background_label.place(relx=0.5, rely=0.5, anchor="center")
root.geometry(f"{new_width}x{new_height}")


# Predict button
predict_button_image = Image.open("images/predict button.png")
button_width = int(predict_button_image.width * 0.7) 
button_height = int(predict_button_image.height * 0.7)  
predict_button_image = predict_button_image.resize((button_width, button_height))
predict_button_image_tk = ImageTk.PhotoImage(predict_button_image)
predict_button = tk.Button(root, image=predict_button_image_tk, command=lambda: predict_click(root,predict_button), borderwidth=0, highlightthickness=0)
predict_button.place(relx=0.5, rely=0.6, anchor="center")

#BUILDING THE MODEL
built_model=mc(root)
print("model built success")


root.mainloop()