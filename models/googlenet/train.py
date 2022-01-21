import numpy as np
import tensorflow as tf
from googlenet_model import GoogLeNet

from models.googlenet.model import GoogLeNet
from utils.load_data import load_data


IMAGE_FOLDER_SPLIT_BY_CLASS = 'data/in/train_by_folder'
SAVE_WEIGHT = 'data/in/weight/googlenet2-6.model'
LOAD_WEIGHT = 'data/in/weight/googlenet2-6.model'

def init_model(): 
    model = GoogLeNet()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])
    return model 

def train_model():
    model = init_model()
    # model.load_weights(LOAD_WEIGHT)
    (X_train, y_train), (X_test, y_test) = load_data(IMAGE_FOLDER_SPLIT_BY_CLASS)
    model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=10,
        verbose=1,
    )
    prediction = model.predict(X_test)

    cce = tf.keras.losses.CategoricalCrossentropy()
    print("val_loss_1", cce(y_test, prediction[0]).numpy())
    print("val_loss_2", cce(y_test, prediction[1]).numpy())
    print("val_loss_3", cce(y_test, prediction[2]).numpy())

    m = tf.keras.metrics.Accuracy()
    m.update_state(y_true=y_test,  y_pred=prediction[2])
    print("val_accuracy_3", m.result().numpy())

    # model.save(SAVE_WEIGHT)
    return model


def load_model():
    model = init_model()
    return model.load_weights(LOAD_WEIGHT)
