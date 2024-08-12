import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import time
import os

import argparse
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
model_vietocr = Predictor(config)

def perform_vietocr(image):
    prediction_text, conf = model_vietocr.predict(image, return_prob=True)

    return prediction_text, conf