from ensure import ensure_annotations
from pathlib import Path
from box import ConfigBox
import yaml
from cnnClassifier import logger
from Exception_file.exception import CustomException
import sys
import os
import json
from typing import Any
import joblib
import base64





@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox : 
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'yaml file from path {path_to_yaml} loaded successfully...!')
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)
    

@ensure_annotations
def create_directories(path_to_directories : list, verbose = True) : 
    for path in path_to_directories:
        os.makedirs(path, exist_ok= True)
        if verbose:
            logger.info(f'created directory at : {path}')


@ensure_annotations
def save_json(path : Path, data : dict):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    logger.info(f'saved json file at : {path}')


@ensure_annotations
def load_json(path : Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f'json file loaded successfully from : {path}')
    return ConfigBox(content)


@ensure_annotations
def save_bin(data : Any, path : Path) -> Any:
    joblib.dump(value=data, filename=path)
    logger.info(f'binary file saved at : {path}')


@ensure_annotations
def load_bin(path : Path) -> Any:

    data = joblib.load(path)
    logger.info(f'binary file loaded from : {path}')
    return data


@ensure_annotations
def get_size(path : Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f'---{size_in_kb} KB'



def decodeImage(imgstring, fileName) :
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())