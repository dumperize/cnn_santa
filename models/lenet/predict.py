import numpy as np
import pandas as pd
from models.lenet.train import train_model
from utils.get_list_images_predict import get_image_names
from utils.load_data import load_image

BASEPATH = 'data/test/'


def prefict():
    model = train_model()  # load_model
    image_names = get_image_names(BASEPATH)
    images = np.array([load_image(BASEPATH + image) for image in image_names])

    prediction = model.predict(images)

    df = pd.DataFrame({'image_name': image_names,
                      'class_id': [np.argmax(x) for x in prediction],
                       })
    string = df.to_csv(index=False, sep='\t')
    file = open('data/out/submission.csv', 'w')
    file.write(str(string))
    file.close()
