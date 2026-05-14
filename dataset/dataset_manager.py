from functools import partial
import numpy as np
import os
import torch
from .options import dataset_options, transform_options
from torch.utils.data import DataLoader


class DatasetManager():
    def __init__(self, train_bs=128, eval_bs=256, seed=0, n_workers=4, 
                 train_d_type='AudiosetDataset', test_d_type='AudiosetDataset',
                 train_path='data', test_path='data',
                 train_tf_op=None, test_tf_op=None, 
                 **kwargs):

        np.random.seed(seed)
        self.bd_mode = False
        if train_d_type not in dataset_options:
            print(train_d_type)
            raise('Unknown Dataset')
        elif test_d_type not in dataset_options:
            print(test_d_type)
            raise('Unknown Dataset')

        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers
        self.train_path = train_path
        self.test_path = test_path

        try:
            env_n_workers = os.environ['SLURM_CPUS_PER_TASK']
            if env_n_workers is not None:
                self.n_workers = int(env_n_workers)
            print('setting n_workers base on SLURM, n_workers is {}'.format(self.n_workers))
        except:
            print('setting n_workers base on SLURM failed, n_workers is {}'.format(self.n_workers))
        if "override_n_workers" in kwargs:
            self.n_workers = kwargs["override_n_workers"]
            print('override n_workers, n_workers is {}'.format(self.n_workers))

        train_tf = transform_options[train_tf_op]["train_transform"]
        test_tf = transform_options[test_tf_op]["test_transform"]
        self.train_tf = train_tf
        self.test_tf = test_tf

        kwargs['seed'] = seed
        kwargs['n_workers'] = self.n_workers
        self.train_set = dataset_options[train_d_type](train_path, train_tf, False, kwargs)
        self.test_set = dataset_options[test_d_type](test_path, test_tf, True, kwargs)
        self.train_set_build_fn = partial(dataset_options[train_d_type], train_path, train_tf, False)
            
    def get_loader(self, train_shuffle=True, drop_last=False, train_sampler=None, test_sampler=None, persistent_workers=True, pin_memory=True):
        if train_shuffle is False or train_sampler is None:
            train_loader = DataLoader(
                dataset=self.train_set, pin_memory=pin_memory, persistent_workers=persistent_workers,
                batch_size=self.train_bs, drop_last=drop_last,
                num_workers=self.n_workers, shuffle=train_shuffle,
            )
            test_loader = DataLoader(
                dataset=self.test_set, pin_memory=pin_memory, persistent_workers=persistent_workers,
                batch_size=self.train_bs, drop_last=False,
                num_workers=self.n_workers, shuffle=False,
            )
        else:
            train_loader = DataLoader(
                dataset=self.train_set, pin_memory=pin_memory, persistent_workers=persistent_workers,
                batch_size=self.train_bs, drop_last=drop_last, 
                num_workers=self.n_workers, sampler=train_sampler,
            )
            test_loader = DataLoader(
                dataset=self.test_set, pin_memory=pin_memory, persistent_workers=persistent_workers,
                batch_size=self.eval_bs, drop_last=False,
                num_workers=self.n_workers, sampler=test_sampler
            )
        return train_loader, test_loader
    


class RayCollateFunction(torch.nn.Module):
    def __init__(self, **kwargs):
        super(RayCollateFunction, self).__init__()

    def forward(self, batch):
        # Ray may hand out read-only numpy buffers; copy to keep Torch happy.
        x_0 = torch.from_numpy(np.array(batch['x_0'], copy=True))
        x_1 = torch.from_numpy(np.array(batch['x_1'], copy=True))
        del batch
        return x_0, x_1
    
    
class RayDatasetManager(DatasetManager):
    def __init__(self, train_bs=128, eval_bs=256, seed=0, n_workers=4, 
                 train_d_type='AudiosetDataset', test_d_type='AudiosetDataset',
                 train_path='data', test_path='data',
                 train_tf_op=None, test_tf_op=None, **kwargs):
        super(RayDatasetManager, self).__init__(
            train_bs=train_bs, eval_bs=eval_bs, seed=seed, n_workers=n_workers,
            train_d_type=train_d_type, test_d_type=test_d_type, train_path=train_path,
            test_path=test_path, train_tf_op=train_tf_op, test_tf_op=test_tf_op, **kwargs
        )
        # self.prefetch_batches = 2

    def _build_train_set(self):
        self.train_set = self.train_set_build_fn({})

    def get_loader(self, **kwargs):
        train_loader = self.train_set.iter_torch_batches(
            batch_size=self.train_bs, drop_last=True, collate_fn=RayCollateFunction(),
            # prefetch_batches=self.prefetch_batches,
            # local_shuffle_buffer_size=self.train_bs * self.prefetch_batches,
        )
        return train_loader, None
    
    

