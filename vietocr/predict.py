
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from tool.predictor import Predictor
from tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)
    # print(config)
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    detector = Predictor(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':

    # main()
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    print(config)
    detector = Predictor(config)

    img = '/dataset_sequence/99796.jpg'
    img = Image.open(img)
    plt.imshow(img)
    s = detector.predict(img)
    print(s)
