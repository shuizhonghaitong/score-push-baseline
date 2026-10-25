# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:01:23 2023

@author: enrico
"""
import streamlit as st
import numpy as np
import pandas as pd
# import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# import h5py
import tensorflow as tf

from PIL import Image as PILImage

# Set page title
st.title("Soil image to particle size distribution curve")
st.write("developed by [Enrico Soranzo](mailto:enrico.soranzo@boku.ac.at)")
st.image('BOKU.png', width=100)
st.image('GRID_no_BG.png', width=100)
st.write("Funded by the European Union under the MSCA Staff Exchanges project 101182689 [GRID](https://grid.boku.ac.at)")
st.write('The model was trained and validated on pictures taken with the following smartphones (resolutions):')
st.markdown('- Motorola Edge (1800x4000)')
st.markdown('- Samsung A52 (6936x9248)')
st.write('The following smartphones (resolutions) are also supported but not validated:')
st.markdown('- iPhone 14 (3024x4032)')
st.write('Use with other devices or resolutions may not yield optimal outcome.')
st.write('Please contact the author if you want to extend the support to a new smartphone.')
st.title("Reference")
st.write(f"[Convolutional neural network prediction of the particle size distribution of soil from close-range images]({'https://doi.org/10.1016/j.sandf.2025.101575'})")

# Upload an image
uploaded_image = st.file_uploader("Upload a soil image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:

    # Check if image dimensions are larger than 1600x720
    image = PILImage.open(uploaded_image)
    width, height = image.size

    scale = 1

    # Motorola Edge
    if width == 1800 or width == 4000:
        # Calculate the scaling factor
        scale = 720/1800

    # Samsung A52
    if width == 6936 or width == 9248:
        # Calculate the scaling factor
        scale = 1600/9248

    # HPC mobile phone
    if width == 3024 or width == 4032:
        # Calculate the scaling factor
        scale = 552/1673
        
    # Calculate the new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
            
    # Resize the image
    resized_image = image.resize((new_width, new_height),PILImage.Resampling.LANCZOS)       

    # Crop the image
    
    # Define cropping size
    width_crop = 400
    height_crop = 400   
    
    # Check if the image dimensions are already cropped
    if resized_image.size != (width_crop, height_crop):
        
        # Calculate the cropping coordinates to get a square image
        width, height = resized_image.width, resized_image.height
        left = (width - width_crop) // 2
        upper = (height - height_crop) // 2
        right = left + width_crop
        lower = upper + height_crop

        cropped_image = resized_image.crop((left, upper, right, lower))
    else: 
        cropped_image = resized_image    
            
    # Display the cropped image
    st.image(cropped_image, caption="Resized and cropped image", use_column_width=True)

    # Split channels
    red_channel, green_channel, blue_channel = cropped_image.split()
    # Import channels and normalise
    red_array = np.array(red_channel)
    green_array = np.array(green_channel)
    blue_array = np.array(blue_channel)
    # Convert to dataframe
    red_df = pd.DataFrame(red_array)
    green_df = pd.DataFrame(green_array)
    blue_df = pd.DataFrame(blue_array)        
    # Convert to a list
    red_list = red_df.values.tolist()
    green_list = green_df.values.tolist()
    blue_list = blue_df.values.tolist()

    # Create input
    X = []   
    X.append([red_list,green_list,blue_list])
            
    # Scale for mobilenet
    X0 = np.array(X)
    X = tf.keras.applications.mobilenet.preprocess_input(X0)
    # Transponse data
    X = tf.transpose(X, perm=[0, 2, 3, 1])
          
    # Load your pre-trained regression model
    model = tf.keras.models.load_model('CNN.h5')  # Load your regression model
    
    # Predict parameters
    predictions = model.predict(X)
    
    # Parameter b
    predictions_b = []
    for i in predictions[0]:
        predictions_b.append(i)

    # Parameter c
    predictions_c = []
    for i in predictions[1]:
        predictions_c.append(i)

    # Plot the curve
    st.write("Predicted particle size distribution")
    fig = plt.figure(dpi=200,figsize=(5, 3))
    ax = fig.add_subplot(1,1,1)
    plt.grid(which='major', linewidth=0.5)
    plt.grid(which='minor', linewidth=0.25)
    plt.semilogx()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.xticks([0.001,0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200],labels=[0.001,0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200],fontsize=6)
    plt.xlim([0.001,200])
    plt.ylim([0,1])
    plt.xlabel('Particle size $d$ (mm)')
    plt.ylabel('Finer')
    plt.plot([0.002,0.002],[0,1],c='black',linewidth=0.75)
    plt.plot([0.063,0.063],[0,1],c='black',linewidth=0.75)
    plt.plot([2,2],[0,1],c='black',linewidth=0.75)
    plt.plot([63,63],[0,1],c='black',linewidth=0.75)
    plt.text(0.0012,1.03,'Cl')
    plt.text(0.01,1.03,'Si')
    plt.text(0.27,1.03,'Sa')
    plt.text(10,1.03,'Gr')
    plt.text(100,1.03,'Bo')
    
    b = math.exp(predictions_b[0])
    c = predictions_c[0]
    
    plt.text(0.0012,0.94,'$y = 1 - \exp(-(d/b)^c)$')
    plt.text(0.0012,0.86,'b = ' + str(np.round(b,3)))
    plt.text(0.0012,0.78,'c = ' + str(np.round(c[0],3)))
    
    x = np.logspace(-3,3, num = 100)
    y = []
    
    # Fitted
    for j in x:
        y.append(1-math.exp(-(j/b)**c))     
    plt.plot(x,y,linewidth=1,c='tab:blue')

    
    st.pyplot(fig)

