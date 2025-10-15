import os
import math
import numpy as np
import zipfile
import shutil
import gdown
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.Classification import logger
from src.Classification.entity.config_entity import (DataIngestionConfig)
from src.Classification.constants import *
from src.Classification.utils.common import read_yaml, write_yaml


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    
    def download_file(self)-> str:  
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            if self.config.is_ggdriver:
                file_id = dataset_url.split("/")[-2]
                prefix = 'https://drive.google.com/uc?/export=download&id='
                gdown.download(prefix+file_id, zip_download_dir)
            else:
                shutil.copy(dataset_url, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        # Remove file .zip
        try:
            os.remove(self.config.local_data_file)
            logger.info(f"File {self.config.local_data_file} has been deleted successfully.")
        except Exception as e:
            logger.error(f"Error occurred while deleting {self.config.local_data_file}: {e}")


    def split_dataset(self):
        """
        Splip data to Train/Validation/Test
        """
        try:
            os.makedirs(self.config.dataset_dir, exist_ok=True)
            data_dir = os.path.join(self.config.unzip_dir, "DefectHole")
            classes = os.listdir(data_dir)
            logger.info(f"Catalog: {classes}")

            # Set Class number in Params
            params = read_yaml(PARAMS_FILE_PATH)
            params.CLASSES = len(classes)
            write_yaml(PARAMS_FILE_PATH, params)

            ## Show Dataset Count
            counts = []
            for i in range(len(classes)):
                class_dir = data_dir + '/' + classes[i]
                count = len([name for name in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, name))])
                counts.append(count)
            logger.info(f"Count: {counts}")
            x = np.array(classes)
            y = np.array(counts)
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.barh(x,y)
            plt.title("Count Datasets by Class")
            plt.savefig(os.path.join(self.config.root_dir, "CountDatasets.png"))

            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(self.config.dataset_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                for class_name in classes:
                    os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
            
            train_ratio = self.config.params_train_val_test_ratio[0]
            val_ratio = self.config.params_train_val_test_ratio[1]
            test_ratio = self.config.params_train_val_test_ratio[2]

            if (train_ratio == 0 or val_ratio == 0 or test_ratio == 0):
                logger.error("Ratio Value Error! Non of the ratios can be 0")
                raise Exception("Ratio Value Error! Non of the ratios can be 0")
            
            if (math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-9) == False):
                logger.error("Ratio Value Error! Expected sum of ratios = 1")
                raise Exception("Ratio Value Error! Expected sum of ratios = 1")

            logger.info("Splitting dataset -> Train/Validation/Test")
            sizeof_dataset = 0
            for class_name in classes:
                source_dir = os.path.join(data_dir, class_name)
                images = os.listdir(source_dir)
                sizeof_dataset = sizeof_dataset + len(images)
                train, temp = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
                val, test = train_test_split(temp, test_size=(1 - val_ratio / (val_ratio + test_ratio)), random_state=42)
                
                for split, split_images in zip(['train', 'val', 'test'], [train, val, test]):
                    for image in split_images:
                        src_path = os.path.join(source_dir, image)
                        dest_path = os.path.join(self.config.dataset_dir, split, class_name, image)
                        shutil.copy(src_path, dest_path)
            logger.info("Splitted dataset -> Done")

            logger.info(f"Size of dataset: {sizeof_dataset} images")
            sizeof_train = int(sizeof_dataset*train_ratio)
            sizeof_val = int(sizeof_dataset*val_ratio)
            sizeof_test = sizeof_dataset - sizeof_train - sizeof_val
            logger.info(f"Train dataset: {sizeof_train} images")
            logger.info(f"Validation dataset: {sizeof_val} images")
            logger.info(f"Test dataset: {sizeof_test} images")
            
            # Zip dataset
            shutil.make_archive(self.config.dataset_dir, 'zip', self.config.dataset_dir)
            
        except Exception as e:
            raise e

