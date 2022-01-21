import numpy as np
import pandas as pd
from models.googlenet.train import train_model
from utils.get_list_images_predict import get_image_names
from utils.load_data import load_image

BASEPATH = 'data/test/'


def prefict():
    model_google = train_model()  # load_model
    image_names = get_image_names(BASEPATH)
    images = np.array([load_image(BASEPATH + image) for image in image_names])

    prediction = model_google.predict(images)

    prediction = np.dstack((
        np.argmax(prediction[0], axis=1),
        np.argmax(prediction[1], axis=1),
        np.argmax(prediction[2], axis=1),
    ))

    df = pd.DataFrame({'image_name': image_names,
                      'class_id': [np.argmax(np.bincount(x)) for x in prediction[0]]})
    string = df.to_csv(index=False, sep='\t')
    file = open('data/out/submission.csv', 'w')
    file.write(str(string))
    file.close()
