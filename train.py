import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

import audio_utils
from DataManager import data_manager_dns_challenge, data_manager_wsjcam0, data_manager_l3das22, data_manager
from hparams import hparams
from utils import print_to_file
from torch.cuda.amp import autocast, GradScaler

# from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate

from utility import sdr, pcm_loss
import DeFTAN_AA
from ptflops import get_model_complexity_info
from thop import profile, clever_format
import random
# import test

# Wrapper class to run PyTorch model
class Runner(object):
    def __init__(self, hparams):
        # TODO: model, criterion
        self.model = DeFTAN_AA.Network()

        self.criterion = pcm_loss.PCM_Loss()

        self.criterion_eval = sdr.negative_SDR()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-4)

        # summary
        self.writer = SummaryWriter(logdir=hparams.logdir)

        # save hyperparameters
        path_hparam = Path(self.writer.logdir, 'hparams.txt')
        if not path_hparam.exists():
            print_to_file(path_hparam, hparams.print_params)

    # Running model for train, test and validation.
    def run(self, dataloader, mode: str, epoch: int):
        self.model.train() if mode == 'train' else self.model.eval()
        self.scaler = torch.cuda.amp.GradScaler()

        avg_loss = 0.
        avg_eval = 0.
        pbar = tqdm(dataloader, desc=f'{mode} {epoch:3d}', dynamic_ncols=True)

        for i_batch, (x_, y_) in enumerate(pbar):
            # data
            x = x_[0].to('cuda')
            y = y_[0].to('cuda')
            with autocast(enabled=True):
                if mode == 'train':
                    out = self.model(x)
                    loss = self.criterion(x[:, 0], out, y)
                else:
                    with torch.no_grad():
                        out = self.model(x)
                        loss = self.criterion(x[:, 0], out, y)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ')
                exit(1)

            # evaluation
            with torch.no_grad():
                eval_result = self.criterion_eval(out, y)
                # eval_result = si_sdr(out, y).mean()

            if mode == 'train':
                # if autocast == True
                self.model.zero_grad()
                self.scaler.scale(loss.mean()).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # if autocast == False
                # self.optimizer.zero_grad()  # make all gradients zero
                # loss.mean().backward()  # calculate all gradients
                # self.optimizer.step()  # update parameters using gradients

                loss = loss.item()
            elif mode == 'valid':
                loss = loss.item()
            else:
                pass

            avg_loss += loss
            avg_eval += eval_result

        avg_loss = avg_loss / len(dataloader.dataset)
        avg_eval = avg_eval.item() / len(dataloader.dataset)

        if mode == 'train':
            print(avg_eval * len(hparams.device))
        else:
            print(avg_eval)

        return avg_loss, avg_eval


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)

def cleanup():
    torch.distributed.destroy_process_group()

def main_worker(rank, world_size):
    # set rendom seed to fixed variable
    random_seed = 365
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # Activate this line when you use multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)

    # set up DDP and cuda
    setup(rank, world_size)
    rank = hparams.out_device + rank
    torch.cuda.set_device(rank)

    # get dataloader    # TODO: modify data_manager.py to load data properly
    if hparams.dataset_name == 'wsjcam0':
        train_loader, valid_loader, test_loader, train_sampler = data_manager_wsjcam0.get_dataloader(hparams)
    elif hparams.dataset_name == 'dns_challenge':
        train_loader, valid_loader, test_loader, train_sampler = data_manager_dns_challenge.get_dataloader(hparams)
    elif hparams.dataset_name == 'l3das22':
        train_loader, valid_loader, test_loader, train_sampler = data_manager_l3das22.get_dataloader(hparams)
    elif hparams.dataset_name == 'chime3':
        train_loader, valid_loader, test_loader, train_sampler = data_manager.get_dataloader(hparams)
    else:
        ValueError

    # Runner object
    runner = Runner(hparams)
    # set model to use DDP
    runner.model.cuda(rank)
    runner.criterion.cuda(rank)
    runner.model = torch.nn.parallel.DistributedDataParallel(runner.model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        total_params = sum(p.numel() for p in runner.model.parameters())    # calculate the number of total parameters
        runner.writer.add_text('Text', 'Parameter size : ' + str(total_params), 0)
        runner.writer.add_text('Text', 'Batch size : ' + str(hparams.batch_size), 0)
        print(f'total params = {total_params}')
        print(f'batch size = {hparams.batch_size}')
        print(f'logdir = {hparams.logdir}')

        # MAC=2FLOPs
        with torch.cuda.device(0):
            net = DeFTAN_AA.Network()
            macs, params = get_model_complexity_info(net, (4, 64000), as_strings=True, print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # # set the layout of custom scalars
        # dict_custom_scalars = dict(loss=['Multiline', ['loss/train', 'loss/valid']],        # layout of custom scalars
        #                            eval=['Multiline', ['eval/train', 'eval/valid']])
        # runner.writer.add_custom_scalars(dict(training=dict_custom_scalars))

    # train and valid
    for epoch in range(hparams.num_epochs):
        # train
        train_sampler.set_epoch(epoch)
        train_loss, train_eval = runner.run(train_loader, 'train', epoch)

        if rank == 0:
            runner.writer.add_scalar('loss/train', train_loss, epoch)
            runner.writer.add_scalar('eval/train', train_eval, epoch)

            # validation
            valid_loss, valid_eval = runner.run(valid_loader, 'valid', epoch)
            runner.writer.add_scalar('loss/valid', valid_loss, epoch)
            runner.writer.add_scalar('eval/valid', valid_eval, epoch)
            # scheduler
            runner.scheduler.step(valid_loss)
            print("lr:", runner.optimizer.param_groups[0]['lr'])

            # saving the parameters
            state_dict = runner.model.state_dict()

            if epoch == 0:
                min_valid_loss = valid_loss
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss
                torch.save(state_dict, Path(runner.writer.logdir, 'max.pt'))
            # two stage approach
            torch.save(state_dict, Path(runner.writer.logdir, 'max.pt'))
            with open(hparams.logdir + '/loss.txt', 'a') as f:    # writing loss value for each epoch
                f.write(str(epoch) + ': ' + str(float(valid_eval)) + '\n')

    print("Train Finished")

    runner.writer.close()


def main():
    ngpus_per_node = len(hparams.device)
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,), join=True)


if __name__ == '__main__':
    # check overwrite or not
    if list(Path(hparams.logdir).glob('events.out.tfevents.*')):
        while True:
            s = input(f'"{hparams.logdir}" already has tfevents. continue? (y/n)\n')
            if s.lower() == 'y':
                shutil.rmtree(hparams.logdir)
                os.makedirs(hparams.logdir)
                break
            elif s.lower() == 'n':
                exit()

    main()
