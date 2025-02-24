import os
import argparse
import importlib
import cv2 as cv
import torch
import torch.backends.cudnn
import random
import numpy as np
import admin.settings as ws_settings
from shutil import copyfile
from datetime import date


def run_training(train_module, train_name, seed, cudnn_benchmark=True,custom_setting = None):
    """Run a train scripts in train_settings.
    args:
        train_module: Name of module in the "train_settings/" folder.
        train_name: Name of the train settings file.
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """

    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    # dd/mm/YY
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    print('Training:  {}  {}\nDate: {}'.format(train_module, train_name, d1))

    settings = ws_settings.Settings()
    settings.module_name = train_module
    settings.script_name = train_name
    settings.project_path = 'train_settings/{}/{}'.format(train_module, train_name)

    settings.seed = seed

    if custom_setting is not None:
        checkpoint = custom_setting.checkpoint
        exp_name = custom_setting.exp_name
        settings.single_cls = custom_setting.single_cls
        settings.pre_crop = custom_setting.pre_crop
    else:
        checkpoint = None
        exp_name = ''
        settings.single_cls = None
        settings.pre_crop = None

    # pretrained checkpoint path
    settings.checkpoint_path = f'pre_trained_models/PWarpCSFNet_{checkpoint}_pfpascal.pth.tar' \
        if checkpoint else None

    # will save the checkpoints there
    folder = settings.project_path + '_' + checkpoint if checkpoint else settings.project_path
    folder += '_' + exp_name

    save_dir = os.path.join(settings.env.workspace_dir, folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(settings.project_path + '.py', os.path.join(save_dir, settings.script_name + '.py'))

    settings.project_path += '_' + checkpoint if checkpoint else ''
    settings.project_path += '_' + exp_name

    expr_module = importlib.import_module('train_settings.{}.{}'.format(train_module.replace('/', '.'),
                                                                        train_name.replace('/', '.')))
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--seed', type=int, default=1992, help='Pseudo-RNG seed')
    parser.add_argument('--checkpoint', default = None, choices = ['SS','WS',None],
                        help = 'which checkpoint to load')
    parser.add_argument('--exp_name', type = str, default = '')
    parser.add_argument('--single_cls', type=str, default=None)
    parser.add_argument('--pre_crop', default = False, action='store_true')
    args = parser.parse_args()

    # args.seed = random.randint(0, 3000000)
    args.seed = torch.initial_seed() & (2 ** 32 - 1)
    print('Seed is {}'.format(args.seed))
    random.seed(int(args.seed))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    run_training(args.train_module, args.train_name, cudnn_benchmark=args.cudnn_benchmark,
                 seed=args.seed, custom_setting=args)


if __name__ == '__main__':
    main()