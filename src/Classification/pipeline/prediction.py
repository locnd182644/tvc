import os 
import numpy as np
from src.Classification.constants import *
from src.Classification.utils.common import read_yaml
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.params = read_yaml(PARAMS_FILE_PATH)

    def predict(self):
        model = load_model(os.path.join("artifacts", "training", "model.h5"))
        image_name = self.filename

        test_image = image.load_img(image_name, target_size=self.params.IMAGE_SIZE)
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)
        predictions = model.predict(test_image)

        data_dir = os.path.join("artifacts", "data_ingestion", "DefectHole")
        classes = os.listdir(data_dir)

        result = {classes[i]: float(predictions[0][i]) for i in range(len(classes))}
        return result

