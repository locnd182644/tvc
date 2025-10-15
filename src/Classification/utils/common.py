import os
from box.exceptions import BoxValueError
import yaml
from src.Classification import logger
from src.Classification.constants import *
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            print(content)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def write_yaml(path_to_yaml: Path, config: ConfigBox):
    """write yaml file from ConfigBox
    """
    try:
        config_dict = config.to_dict()
        with open(path_to_yaml, 'w') as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False, allow_unicode=True)
            logger.info(f"yaml file: {path_to_yaml} stored successfully")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
    
import re
def get_local_version():
    """
    Get the current version number.

    Returns:
        str: The current version number.
    """
    with open(VERSION_FILE_PATH, "r") as f:
        content = f.read()

    match = re.search(r'(\d+)\.(\d+)\.(\d+)', content)
    if match:
        return match.group()
    return "0.0.0"

def update_number_local_version(type: str):
    """
    Update the version number based on the type of version change.

    Args:
        type (str): The type of version change. Can be 'major', 'minor', or 'patch'.

    Raises:
        ValueError: If the type is not 'major', 'minor', or 'patch'.
    """
    try:
        # Read current version
        with open(VERSION_FILE_PATH, "r") as f:
            content = f.read()

        # Tìm version
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', content)
        if match:
            major, minor, patch = map(int, match.groups())
            
            if type == "major":
                major += 1
                minor = 0
                patch = 0
            elif type == "minor":
                minor += 1
                patch = 0
            elif type == "patch":
                patch += 1
            else:
                raise ValueError("Invalid version type. Choose 'major', 'minor', or 'patch'.")

            new_version = f"{major}.{minor}.{patch}"

            # Ghi lại version mới
            with open("version.py", "w") as f:
                f.write(f'__version__ = "{new_version}"')

            logger.info(f"Updated version: {new_version}")
        return new_version
    
    except ValueError as ve:
        logger.error(f"Error updating version: {ve}")    