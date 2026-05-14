import argparse
import torch
import mlconfig
import dataset
import models
import losses
import util
import misc
import os
import sys
import numpy as np
import time
import math
from exp_mgmt import ExperimentManager
from engine_finetune import train_epoch, evaluate
from timm.optim import optim_factory

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


parser = argparse.ArgumentParser(description='AudioMosaic')
# General Options
parser.add_argument('--seed', type=int, default=7, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
# distributed training parameters
parser.add_argument('--ddp', action='store_true', default=False)
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
# Debugging Options
parser.add_argument('--type', type=str, default="ata", choices=["ata", "tta"], help='type of evaluation dataset')


def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config    
    
    # Prepare Model
    model = config.model().to(device)
    exp.load_state(model, 'model_state_dict_epochbest', strict=True)
    model = model.eval()

    # Prepare data 
    logger.info("Best Epoch loaded for evaluation.")
    data_path = "dataset/envsdd_util/" + args.type + "/test"
    for test_split in ["test01.json", "test02.json", "test03.json", "test04.json"]:
        test_data = dataset.audioset.AudiosetDataset(
            dataset_json_file=os.path.join(data_path, test_split),
            audio_conf=dataset.options.transform_options["EnvSDDFinetune"]["test_transform"],
            roll_mag_aug=False,
            mode='eval',
            label_csv="dataset/envsdd_util/class_labels_indices.csv"
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
        )
        eval_stats = evaluate(model, test_loader)
        payload = "{} EER: {:.4f} AUC: {:.4f} from {} files".format(test_split, eval_stats['EER'], eval_stats['AUC'], eval_stats['N'])
        logger.info('\033[33m'+payload+'\033[0m')

    # Prepare Model
    model = config.model().to(device)
    exp.load_state(model, 'model_state_dict', strict=True)
    model = model.eval()

    # Prepare data 
    logger.info("Last Epoch loaded for evaluation.")
    data_path = "dataset/envsdd_util/" + args.type + "/test"
    for test_split in ["test01.json", "test02.json", "test03.json", "test04.json"]:
        test_data = dataset.audioset.AudiosetDataset(
            dataset_json_file=os.path.join(data_path, test_split),
            audio_conf=dataset.options.transform_options["EnvSDDFinetune"]["test_transform"],
            roll_mag_aug=False,
            mode='eval',
            label_csv="dataset/envsdd_util/class_labels_indices.csv"
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
        )
        eval_stats = evaluate(model, test_loader)
        payload = "{} EER: {:.4f} AUC: {:.4f} from {} files".format(test_split, eval_stats['EER'], eval_stats['AUC'], eval_stats['N'])
        logger.info('\033[33m'+payload+'\033[0m')
    return 


if __name__ == '__main__':
    global exp, seed
    args = parser.parse_args()
    if args.ddp:
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)
        seed = args.seed
    args.gpu = device
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(
        exp_name=args.exp_name, exp_path=args.exp_path,
        config_file_path=config_filename, args=args,
        enable_online_log=False, eval_mode=True,
    )
    
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        try:
            logger.info('SLURM_NODELIST: {}'.format(os.environ['SLURM_NODELIST']))
        except:
            pass
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)
    if args.ddp: 
        misc.destroy_process_group()
