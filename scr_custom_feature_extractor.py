"""
A simple python script to extract features using deep learning models that are made available alongside pre-trained weights.

Author: Cristiano Patrício
E-mail: cristiano.patricio@ubi.pt
University of Beira Interior, Portugal
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
import argparse
import time
from keras.utils import plot_model
import glob
import numpy as np
import pickle as cPickle
import bz2
import datetime


#########################################################################
#   ARGUMENTS
#########################################################################

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--directory', required=True,
                help='Data input directory')
ap.add_argument('-m', '--model', required=True,
                help='Model: {resnet50, vgg16, vgg19, inception_v3, efficient_net_b0, nas_large}')
ap.add_argument('-o', '--output_dir',
                required=True, help='Directory to save the output file')
ap.add_argument('-t', '--file_type',
                required=True, help='File type: {txt, pkl, pbz2}')
args = vars(ap.parse_args())

directory = args['directory']
model_name = args['model']
output_dir = args['output_dir']
filetype = args['file_type']

# Load pre-trained Model
print("[INFO]: Loading pre-trained model...")

if model_name == "resnet50":
    model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
elif model_name == "vgg16":

    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc1').output)
elif model_name == "vgg19":

    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc1').output)
elif model_name == "inception_v3":

    model = InceptionV3(weights='imagenet', include_top=False, pooling="avg")
elif model_name == "efficient_net_b0":

    model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg")
elif model_name == "nas_large":
    
    model = tf.keras.applications.NASNetLarge(
        weights="imagenet",
        include_top=False,
        pooling="avg")

# List to save extracted features
feature_list = []

# Images path directory
img_path = directory

# Get directories under the directory 'images'
dirs = sorted(os.listdir(img_path))

tic = time.time()

for dir in dirs:
    # Get images under each subdir
    imgs_dir = sorted(os.listdir(os.path.join(img_path, dir)))
    print("[INFO]: Processing images under " +
          str(os.path.join(img_path, dir)) + "...")

    for fname in imgs_dir:

        img = image.load_img(os.path.join(
            img_path, dir, fname), target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        feature = model.predict(img_data)
        feature_np = np.array(feature)
        feature_list.append(feature_np.flatten())

feature_list = np.asarray(feature_list)

# Save extracted features to .PKL, .PBZ2 pickle or .TXT file
output_filename = "features-"+str(model_name)+"-"+str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

if filetype == "txt":
    np.savetxt(os.path.join(output_dir, output_filename+".txt"), feature_list)
elif filetype == "pkl":
    with open(os.path.join(output_dir, output_filename+".pkl"), "wb") as f:
        cPickle.dump(feature_list, f)
elif filetype == "pbz2":
    with bz2.BZ2File(os.path.join(output_dir, output_filename+".pbz2"), "w") as f:
        cPickle.dump(feature_list, f)

print("[INFO]: Extracted features saved at", output_dir)

toc = time.time()

# Print elapsed time
print("[INFO]: Process completed in %.2f minutes." % ((toc-tic)/60))