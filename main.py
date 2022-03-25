def main(config):
    if config.semi_set:
        from SCL_SSL import Solver
        print("semi_supervised setting!")
        print("datasets:", config.datasets)
    else:
        from SCL import Solver
        print("unsupervised setting!")
        print("datasets:", config.datasets)

    solver = Solver(config)
    cudnn.benchmark = True
    return solver.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--just_valid', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_1', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_shape', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--theta', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=10.0)
    parser.add_argument('--datasets', type=str, default='wikipedia')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--semi_set', type=bool, default=False)  # for semi-supervised setting
    parser.add_argument('--sup_rate', type=float, default=0.1)  # % labeled data (0-1)
    parser.add_argument('--cuda', type=str, default='2')

    config = parser.parse_args()

    seed = 123
    print('seed: ' + str(seed))
    import numpy as np

    np.random.seed(seed)
    import random as rn

    rn.seed(seed)
    import os

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn

    cudnn.enabled = False

    import nni
    from nni.utils import merge_parameter
    import logging

    logger = logging.getLogger('scl')
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = merge_parameter(config, tuner_params)
    # params = config

    results = main(params)
