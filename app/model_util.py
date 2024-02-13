import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16pp
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19pp
from tensorflow.keras.applications.xception import Xception, preprocess_input as xceptionpp
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inceptionv3pp
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
from glob import glob
from PIL import Image
from io import BytesIO
import base64
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading of truncated images

dog_names = [item[item.rindex('.')+1:-1] for item in sorted(glob("../data/dogImages/train/*/"))]

def path_to_tensor(content):
    try:
        with Image.open(BytesIO(base64.b64decode(content))) as img:
            #img.load()
            img = img.resize((224, 224)) #resize image to 224x224
            img = img.convert('RGB') #convert to 3 channel RGB
            x = img_to_array(img)
            # Normalize the image tensor
            return np.expand_dims(x, axis=0)
    except IOError:
        print(f"Warning: Skipping corrupted image")
        return None

pre_trained_models = {}
transfer_models = {}
preprocessing = {}

#get pre-trained models and corresponding preprocess_input functions
pre_trained_models['VGG16'] = VGG16(weights='imagenet', include_top=False)
preprocessing['VGG16'] = vgg16pp
pre_trained_models['VGG19'] = VGG19(weights='imagenet', include_top=False)
preprocessing['VGG19'] = vgg19pp
pre_trained_models['Xception'] = Xception(weights='imagenet', include_top=False)
preprocessing['Xception'] = xceptionpp
pre_trained_models['InceptionV3'] = InceptionV3(weights='imagenet', include_top=False)
preprocessing['InceptionV3'] = inceptionv3pp

#get best trained models
for network in ['VGG16', 'VGG19', 'InceptionV3', 'Xception']:
    transfer_models[network] = load_model(f'../saved_models/weights.best.{network}.hdf5')

#Function to predict breed
def predict_breed(content, network='VGG16'):
    #get bottleneck features from pre-trained model
    features = pre_trained_models[network].predict(preprocessing[network](path_to_tensor(content)))
    #use trained model to get prediction
    predicted_vector = transfer_models[network].predict(features)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]