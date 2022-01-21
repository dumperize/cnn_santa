import numpy as np
import os         
import cv2              

import tensorflow as tf
from sklearn.model_selection import train_test_split   
from tqdm import tqdm
from imblearn.over_sampling  import SMOTE


class_names = ['bearder', 'fatherfrost', 'santa']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (224, 224)


def load_image(img_path):
    # Open and resize the img
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE) 
    image = image / 255.0
    return image

def load_data(dataset):
    images = []
    labels = []
        
    print("Loading {}".format(dataset))
        
    # Iterate through each folder corresponding to a category
    for folder in os.listdir(dataset):
        label = class_names_label[folder]
 
        # Iterate through each image in our folder
        for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
            # Get the path name of the image
            img_path = os.path.join(os.path.join(dataset, folder), file)
            image = load_image(img_path) 
                
            # Append the image and its corresponding label to the output
            images.append(image)
            labels.append(label)
                
    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')

    labels = tf.keras.utils.to_categorical(labels, 3)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=25)
    
    shape = X_train.shape
    ReX_train = X_train.reshape(shape[0], shape[1] * shape[2] * shape[3])

    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(ReX_train, y_train)

    X_smote = X_smote.reshape(X_smote.shape[0],  shape[1], shape[2], shape[3])
    
    return [(X_smote, y_smote), (X_test, y_test)]
