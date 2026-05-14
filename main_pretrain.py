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
from engine_pretrain import train_epoch
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
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)


def save_model(model, optimizer, epoch=None):
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    if epoch is not None:
        exp.save_state(model, 'model_state_dict_epoch{:d}'.format(epoch))


def main():
    # Set up Experiments
    logger = exp.logger
    config = exp.config

    # Prepare Data
    data_manager = config.data_manager()
    if 'blr' in exp.config:
        if exp.config.blr_scale == 'linear':
            # Linear scaling
            eff_batch_size = exp.config.data_manager.train_bs * misc.get_world_size()
            exp.config.lr = exp.config.blr * eff_batch_size / 256
        else:
            # Square root scaling
            eff_batch_size = exp.config.data_manager.train_bs * misc.get_world_size()
            exp.config.lr = exp.config.blr * math.sqrt(eff_batch_size)
        if misc.get_rank() == 0:
            logger.info('adjusted lr: {:.6f}'.format(exp.config.lr))

    if isinstance(data_manager, dataset.RayDatasetManager):
        if misc.get_rank() == 0:
            logger.info('Using Ray Dataset Manager')
        sampler_train, sampler_val = None, None
    elif args.ddp:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if misc.get_rank() == 0:
            logger.info('World Size {}'.format(num_tasks))
        sampler_train = torch.utils.data.DistributedSampler(
            data_manager.train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(data_manager.test_set) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(data_manager.test_set, num_replicas=num_tasks,
                                                              rank=global_rank, shuffle=True)
            # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(data_manager.test_set)
    else:
        sampler_train = torch.utils.data.RandomSampler(data_manager.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data_manager.test_set)

    loader = data_manager.get_loader(
        drop_last=True, train_shuffle=True, 
        train_sampler=sampler_train, test_sampler=sampler_val,
        pin_memory=False, 
    )
    train_loader, test_loader = loader
    
    # Prepare Model and Loss
    model = config.model().to(device)
    params = optim_factory.param_groups_weight_decay(
        model, config.weight_decay, no_weight_decay_list=["cls_token", "pos_embed", "logit_scale", "logit_bias"]
    )
    print(model, flush=True)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if misc.get_rank() == 0:
        logger.info("Number of params: %.2f M" % (n_parameters / 1.e6))
        
    optimizer = config.optimizer(params)
    criterion = config.criterion()

    if hasattr(exp.config, 'compile') and exp.config.compile:
        torch._dynamo.config.optimize_ddp = False   # disable DDP optimizer
        print('Compiling Model')
        model = torch.compile(model, dynamic=exp.config.compile_dynamic)
        if hasattr(exp.config, "activation_memory_budget"):
            torch._functorch.config.activation_memory_budget = exp.config.activation_memory_budget
            print('Activation Memory Budget: {:.2f}'.format(torch._functorch.config.activation_memory_budget))

    if args.ddp:
        if hasattr(exp.config, 'find_unused_parameters'):
            find_unused_parameters = exp.config.find_unused_parameters
        else:
            find_unused_parameters = False
        if misc.get_rank() == 0:
            logger.info('DDP')
        if 'sync_bn' in exp.config and exp.config.sync_bn:
            if misc.get_rank() == 0:
                logger.info('Sync Batch Norm')
            sync_bn_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            model = torch.nn.parallel.DistributedDataParallel(sync_bn_network, find_unused_parameters=find_unused_parameters, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    global_step = 0
    if hasattr(exp.config, 'amp') and exp.config.amp:
        scaler = torch.amp.GradScaler() 
    else:
        scaler = None
    
    global_step = 0
    for epoch in range(exp.config.epochs):
        if args.ddp and not isinstance(data_manager, dataset.RayDatasetManager):
            train_loader.sampler.set_epoch(epoch)
        elif isinstance(data_manager, dataset.RayDatasetManager):
            data_manager._build_train_set()
            train_loader, _ = data_manager.get_loader(
                drop_last=True, train_shuffle=True, 
                train_sampler=sampler_train, test_sampler=sampler_val,
                pin_memory=False, 
            )
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch {}".format(epoch) + "="*20)

        model.train()
        train_stats = train_epoch(exp, global_step, train_loader, model, optimizer, scaler, criterion, logger, args)
        global_step = train_stats["global_step"]

        # Save Model
        if epoch % exp.config.save_frequency == 0 and epoch > 0:
            if misc.get_rank() == 0:
                exp.save_epoch_stats(epoch=global_step, exp_stats=train_stats)
                save_model(model_without_ddp, optimizer, epoch)
                
    if misc.get_rank() == 0:
        exp.save_epoch_stats(epoch=global_step, exp_stats=train_stats)
        save_model(model_without_ddp, optimizer, epoch)
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
        enable_online_log=True if not args.debug else False,
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
