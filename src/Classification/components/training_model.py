import os
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import onnx
import tf2onnx

from src.Classification.utils.common import create_directories
from src.Classification.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "val"),
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=os.path.join(self.config.training_data, "train"),
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    @staticmethod
    def convert2onnx(path: Path, model: tf.keras.Model):
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        onnx.save(onnx_model, path)

    def show_graph_training(self, data_log):
        # Lấy dữ liệu từ lịch sử huấn luyện
        loss = data_log.history['loss']
        val_loss = data_log.history['val_loss']
        accuracy = data_log.history['accuracy']
        val_accuracy = data_log.history['val_accuracy']

        epochs = range(1, len(loss) + 1)

        # Tạo biểu đồ
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Vẽ loss trên trục y đầu tiên
        ax1.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'b--', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epochs', fontsize=14)
        ax1.set_ylabel('Loss', color='b', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='b')

        # Vẽ accuracy trên trục y thứ hai
        ax2 = ax1.twinx()  # Tạo trục y thứ hai chia sẻ trục x
        ax2.plot(epochs, accuracy, 'r-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracy, 'r--', label='Validation Accuracy', linewidth=2)
        ax2.set_ylabel('Accuracy', color='r', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='r')

        # Thêm chú thích
        fig.tight_layout()
        ax1.legend(loc='center right')
        ax2.legend(loc='center')
        # plt.title('Training and Validation Loss/Accuracy')
        plt.savefig(os.path.join(self.config.root_dir, "TrainingGraph.png"))


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        checkpoint_path = os.path.join(self.config.root_dir, "checkpoint") 
        create_directories([checkpoint_path])

        cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, mode='min')
        cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_path, "best_current_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            save_freq="epoch",
            verbose=0
        )

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=[cb_reduce_lr, cb_checkpoint, TrainingLogger()]
        )

        self.show_graph_training(history)

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        self.convert2onnx(
            path=os.path.join(self.config.root_dir, "model.onnx"), 
            model=self.model
        )

        with open(TRAINING_LOG, "r+") as file:
            file.truncate(0)



TRAINING_LOG = os.path.join("artifacts", "training", "training.log")
class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(TRAINING_LOG, "a") as f:
            f.write(f"{epoch + 1},{logs['loss']:.4f},{logs['accuracy']:.4f},{logs['val_loss']:.4f},{logs['val_accuracy']:.4f}\n")
