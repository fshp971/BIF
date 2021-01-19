import logging
import os
import sys
import numpy as np
import torch

import utils
import experiments


def main():
    args = utils.get_args()

    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    save_name = args.save_name
    if save_name is None:
        save_name = args.exp_type
    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<18}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    try:
        ''' only add code here '''
        if args.exp_name == 'gmm':
            exp_obj = experiments.GMMSimulation(args)

        elif args.exp_name == 'bnn':
            exp_obj = experiments.DeepLearning(args)

        else:
            raise ValueError('unrecognize experiment {}'.format(args.exp_name))

        exp_obj.run()

    except Exception as e:
        logger.exception("Unexpected exception! %s", e)


if __name__ == '__main__':
    main()
