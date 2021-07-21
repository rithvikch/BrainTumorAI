from flask import Flask, render_template, redirect, url_for, request, flash
from PIL import Image
import urllib.request   
import os
from tensorflow.keras import layers
from werkzeug.utils import secure_filename
from glob import glob
from flask.helpers import flash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import keras
from keras.applications.vgg16 import preprocess_input,VGG16
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,Dropout,GlobalAveragePooling2D,Flatten,Input
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
user = ""
UPLOAD_FOLDER = 'static/uploads/'

listo = []
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
filename = ""
boolean = False

def predict_imported_image():
    m = model_cnn.predict(X_test[len(X_test)-1])
    score = tf.nn.softmax(m)
    n = np.argmax(score)
    s = ""
    if (n == 0):
        s = "glioma_tumor"
    elif (n == 1):
        s = "meningioma_tumor"
    elif (n == 2):
        s = "no_tumor"
    else:
        s = "pituitary_tumor"
    return s


@app.route("/")
def homepage():
    
    return render_template("index.html", content=["A brain tumor is a mass or growth of abnormal cells in your brain.", "More than 200,000 cases of brain tumors are reported every year!", "Some brain tumors are noncancerous (non-dangerous), and some brain tumors are cancerous (malignant).", "Brain tumors can begin in your brain (primary brain tumors), or cancer can begin in other parts of your body and spread to your brain (secondary, or metastatic, brain tumors)."])
@app.route("/", methods = ['POST'])
def upload():
    if 'nm' not in request.files:
        flash("No file part")
        return redirect('homepage')
    file = request.files['nm']
    if file.filename == '':
        flash("No image..")
        return redirect('homepage')
    if file:
        filename = secure_filename(file.filename)
        boolean = True
        
        
        

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
        UPLOAD_FOLDER = 'Testing/pituitary_tumor/'

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        

        X_train = []
        y_train = []

        paths = []
        for r, d, f in os.walk("Training"):
            for file in f:
                if '.jpg' in file:
                    paths.append(os.path.join(r, file))
                    end = None
                    y_train.append(r[len("./Training/"): end])

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)
        X_train = train_datagen.flow_from_directory('/Users/rithvikchikoti/Desktop/Flask Tumor AI app/Training',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'categorical')

        X_test = []
        y_test = []

        paths2 = []
        for r, d, f in os.walk("Testing"):
            for file in f:
                if '.jpg' in file:
                    paths2.append(os.path.join(r, file))
                    end = None
                    y_test.append(r[len("./Testing/"):end])
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
        X_test = test_datagen.flow_from_directory('/Users/rithvikchikoti/Desktop/Flask Tumor AI app/Testing',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        model_cnn=Sequential()
        model_cnn.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(64, 64, 3)))
        model_cnn.add(Conv2D(128,(3,3)))
        model_cnn.add(MaxPooling2D((2,2)))
        model_cnn.add(BatchNormalization())
        model_cnn.add(Conv2D(64,(3,3)))
        model_cnn.add(MaxPooling2D((2,2)))
        model_cnn.add(BatchNormalization())
        model_cnn.add(Conv2D(32,(3,3)))
        model_cnn.add(MaxPooling2D((2,2)))
        model_cnn.add(BatchNormalization())
        model_cnn.add(Flatten())
        model_cnn.add(Dense(128,activation='relu'))
        model_cnn.add(Dropout(0.2))
        model_cnn.add(Dense(64,activation='relu'))
        model_cnn.add(Dense(4,activation='softmax'))
        model_cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

        model_cnn.compile(optimizer='adam',
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        model_cnn.fit(x = X_train, validation_data = X_test, epochs = 10)
        predict_imported_image()



    
        
        
        
        flash("Image successfully uploaded and displayed below")
        return render_template("index.html", filename = filename, content=["A brain tumor is a mass or growth of abnormal cells in your brain.", "More than 200,000 cases of brain tumors are reported every year!", "Some brain tumors are noncancerous (non-dangerous), and some brain tumors are cancerous (malignant).", "Brain tumors can begin in your brain (primary brain tumors), or cancer can begin in other parts of your body and spread to your brain (secondary, or metastatic, brain tumors)."], answer = string)
    
    else:
        return redirect("homepage")
@app.route('/display/<filename>')
def display(filename):
    
    return redirect(url_for('static', filename='uploads/'+filename  ),code=301)



if __name__ == "__main__":
    app.run()
    
