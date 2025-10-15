import tensorflow as tf
from pathlib import Path
from datetime import datetime
import os
import dagshub
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.Classification import logger
from src.Classification.entity.config_entity import EvaluationConfig
from src.Classification.utils.common import save_json

SCORE_NAMES = ["loss", "accuracy", "time"]
CONFUSION_MATRIX_IMGPATH = os.path.join("artifacts", "confusion_matrix.png")

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def _test_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.test_generator = test_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "test"),
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self, tries=2):
        self.model = self.load_model(self.config.path_of_model)
        self._test_generator()
        logger.info("Start Evaluate -----------")
        list_scores = []
        try_count = 0
        while (try_count < tries):
            logger.info("----------- Start evaluate {} time-----------".format(try_count + 1))
            start_time = datetime.now()  # Bắt đầu đo thời gian
            scores = self.model.evaluate(self.test_generator)
            end_time = datetime.now()  # Kết thúc đo thời gian
            sample_nums = self.test_generator.samples
            time = (end_time - start_time).total_seconds() / sample_nums
            scores.append(time)
            list_scores.append(scores)
            logger.info("----------- End evaluate {} time-----------".format(try_count + 1))
            try_count = try_count + 1

        # Get Average the evaluation
        np_scores = np.array(list_scores)
        avg_loss = np.mean(np_scores[:,0])
        avg_accuracy = np.mean(np_scores[:,1])
        avg_time = np.mean(np_scores[:,2])
        self.scores = [avg_loss, avg_accuracy, avg_time]
        logger.info("------------- End Evaluate")

        self.save_scores()
        self.save_confusion_matrix()

    def save_scores(self):
        scores = {SCORE_NAMES[0]: self.scores[0], SCORE_NAMES[1]: self.scores[1], SCORE_NAMES[2]: self.scores[2]}
        save_json(path=Path("scores.json"), data=scores)

    def save_confusion_matrix(self):
        data_list = []
        label_list = []
        batch_index = 0

        while batch_index <= self.test_generator.batch_index:
            data = self.test_generator.next()
            data_list.append(data[0])
            label_list.append(data[1])
            batch_index = batch_index + 1

        data = np.vstack(data_list)
        y_test = np.vstack(label_list)
        y_test = np.argmax(y_test, axis=1)

        prediction = self.model.predict(data)

        y_pred = np.argmax(prediction, axis=1)

        # Create a confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        # Plot the confusion matrix
        classes = self.test_generator.class_indices.keys()
        plt.rcParams['font.family'] = 'Malgun Gothic'
        sns.heatmap(cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Prediction',fontsize=12)
        plt.ylabel('Actual',fontsize=12)
        plt.title('Confusion Matrix',fontsize=16)
        plt.savefig(CONFUSION_MATRIX_IMGPATH)


    def log_into_mlflow(self):
        if (self.config.is_remote_log):
            dagshub.init(repo_owner=self.config.repo_owner, repo_name=self.config.repo_name, mlflow=True)
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics({SCORE_NAMES[0]: self.scores[0], SCORE_NAMES[1]: self.scores[1], SCORE_NAMES[2]: self.scores[2]})
                # mlflow.keras.log_model(self.model, "model", registered_model_name="DefectHole_VGG16Model")
