from logging import getLogger
from config import Config
from model import Model
from trainer import Trainer
from utils import init_seed
from logger import init_logger
from utils import data_preparation
import json
import sys


def infer(config):
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    d = json.loads(open('pnelout.json').read())
    test_data, _, _ = data_preparation(config, d['linkedentrelstring'],d['linkedentrelvecs'])
    model = Model(config).to(config['device'])
    trainer = Trainer(config, model)
    test_result = trainer.evaluate_single(test_data, model_file=config['load_experiment'])



if __name__ == '__main__':
    config = Config(config_dict={'test_single': True,'load_experiment': 'saved3/Fire-At-Dec-02-2021_12-47-22.pth'})
    infer(config)
