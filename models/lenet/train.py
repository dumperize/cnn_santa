import tensorflow as tf
from models.lenet.model import Model
from utils.load_data import load_data


IMAGE_FOLDER_SPLIT_BY_CLASS = 'data/in/train_by_folder'
SAVE_PATH_MODEL = 'data/in/weight/lenet.model'
LOAD_PATH_MODEL = 'data/in/weight/lenet.model'


def init_model():
    model = Model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])
    return model


def train_model():
    (X_train, y_train), (X_test, y_test) = load_data(IMAGE_FOLDER_SPLIT_BY_CLASS)

    model = init_model()
    # model.load(LOAD_PATH_MODEL)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'])
    model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=100,
        verbose=1,
        validation_data=(X_test, y_test)
    )
    # model.save(SAVE_PATH_MODEL)
    return model


def load_model():
    model = init_model()
    return model.load(LOAD_PATH_MODEL)
