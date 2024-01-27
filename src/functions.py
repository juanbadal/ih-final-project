import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from PIL import Image
from tensorflow.keras.models import load_model



def load_process_save_image(filename: str,
                       target_size: tuple=(48, 48),
                       dir_path: str='./../data/raw/new_images/',
                       output_path: str='./../data/cleaned/new_images/'):
    '''
    - Takes an image (filename) from dir_path
    - Loads it into the notebook
    - Resizes to 96x96 to fit with the rest of the data
    - Transforms into grayscale
    - Saves the transformed image in output_path
    '''
    # Create output_path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        img_path = os.path.join(dir_path, filename)
        img = Image.open(img_path)

        # Resize image to 48x48
        img_resized = img.resize(target_size)

        # Convert image to grayscale
        img_grayscale = img_resized.convert('L')

        # Save image in output_path
        output_filename = os.path.join(output_path, filename)
        img_grayscale.save(output_filename)

    print(f'{filename} image processed')



    