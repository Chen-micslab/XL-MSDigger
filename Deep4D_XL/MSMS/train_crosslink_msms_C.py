import argparse
import logging
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from dataset.Crosslink_Dataset import Mydata_label
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_crosslink_model import eval_model


def get_args():
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--msms_norm', type=float, default=10)
    parser.add_argument('--validation', type=int, default=1)
    parser.add_argument('--vali_rate', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--load_msms_param_dir', type=str, default='Deep4D_XL/MSMS/checkpoint/msms_c.pth')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--filename', type=str, default='merged_lib')
    parser.add_argument('--sch', type=int, default=0)
    return parser.parse_args()


def get_mask(peptide, length):
    mask = torch.zeros(peptide.size(0), peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0
    return mask


def train(model, device, epochs=10, batch_size=1, lr=0.001, val_percent=0.1, save_mp=True, sch=0,
          traindir=None, validir=None, checkpoint_dir=None, expect_performance=None):
    if validir == None:
        mydata = Mydata_label(traindir)
        n_val = int(len(mydata) * val_percent)
        n_train = len(mydata) - n_val
        train_data, val_data = random_split(mydata, [n_train, n_val])
    else:
        train_data = Mydata_label(traindir)
        val_data = Mydata_label(validir)
        n_train = len(train_data)
        n_val = len(val_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
    print(max, min)
    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_mp}
            Device:          {device.type}
        ''')
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if sch == 0:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=20, T_mult=2, eta_min=0.00000001)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.2)
    best_index = 0
    for epoch in range(epochs):
        model.train()
        local_step = 0
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='peptide') as pbar:
            for batch in train_loader:
                local_step += 1
                peptide1 = batch['peptide1'].to(device=device, dtype=torch.float32)
                peptide2 = batch['peptide2'].to(device=device, dtype=torch.float32)
                pep1_len = batch['len1'].to(device=device, dtype=torch.float32)
                pep2_len = batch['len2'].to(device=device, dtype=torch.float32)
                peptide_msms = batch['peptide_msms'].to(device=device, dtype=torch.float32)
                charge = batch['charge'].to(device=device, dtype=torch.float32)
                mask1 = get_mask(peptide1, pep1_len).to(device=device, dtype=torch.bool)
                mask2 = get_mask(peptide2, pep2_len).to(device=device, dtype=torch.bool)
                norm = args.msms_norm
                peptide_msms = norm * peptide_msms
                peptide_msms_pre = model(pep1=peptide1, pep2=peptide2, src_key_padding_mask_1=mask1, src_key_padding_mask_2=mask2, charge=charge)
                loss_f = nn.MSELoss(reduction='none')
                loss = loss_f(peptide_msms, peptide_msms_pre)
                mask = (peptide_msms != -1 * norm) * (peptide_msms != 0)
                loss = loss.masked_select(mask)
                pbar.set_postfix(**{'loss (batch)': torch.mean(loss)})
                optimizer.zero_grad()
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                pbar.update(peptide1.shape[0])
                global_step += 1
                if global_step % (n_train // (2 * batch_size)) == 0:
                    for name, value in model.named_parameters():
                        name = name.replace('.', '/')
                    mean_cosine, median_cosine = eval_model(model, val_loader, device, norm)
                    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation Mean cosine: {}'.format(mean_cosine))
                    logging.info('Validation Median cosine: {}'.format(median_cosine))
                    if median_cosine > best_index:
                        torch.save(model.state_dict(),
                                   checkpoint_dir + f'model_param_epoch{epoch + 1}global_step{global_step}#median_cosine{median_cosine}.pth')
                        logging.info(f'Checkpoint {epoch + 1}global_step{global_step} saved !')
                        best_index = median_cosine
        scheduler.step()
        if save_mp:
            logging.info('Created checkpoint directory')
            torch.save(model.state_dict(), checkpoint_dir + f'model_param_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()


def do_train(args):
    if args.validation != 0:
        traindir = f'./dataset/data/{args.filename}_train'
        validir = f'./dataset/data/{args.filename}_validation'
    else:
        traindir = f'./dataset/data/{args.filename}'
        validir = None
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    logging.info(f'Using device {device}')
    from model.crosslink_msms_model_cleavable import Transformer as deep_model
    checkpoint_dir = f'./checkpoint/{args.filename}_msms/'
    if os.path.exists(checkpoint_dir):
        pass
    else:
        os.mkdir(checkpoint_dir)
    model = deep_model(feature_len=args.feature_len,
                       d_model=args.d_model,
                       nhead=args.nheads,
                       num_encoder_layers=args.num_encoder_layers,
                       dim_feedforward=args.dim_feedforward,
                       dropout=args.dropout,
                       activation=args.activation)

    logging.info(f'Model:\n'
                 f'\tfeature_len is {args.feature_len}\n'
                 f'\td_model is {args.d_model}\n'
                 f'\targs.nheads is {args.nheads}\n'
                 f'\targs.num_encoder_layers is {args.num_encoder_layers}\n'
                 f'\targs.dim_feedforward is {args.dim_feedforward}\n'
                 f'\targs.dropout is {args.dropout}\n'
                 f'\targs.activation is {args.activation}\n')
    if args.load_msms_param_dir:
        model_state_dict = torch.load(args.load_msms_param_dir, map_location=device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in model_state_dict.items()})
        logging.info(f'Model parameters loaded from {args.load_msms_param_dir}')
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device=device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    train(model=model,
          device=device,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          val_percent=args.vali_rate,
          save_mp=True,
          sch=args.sch,
          traindir=traindir,
          validir=validir,
          checkpoint_dir=checkpoint_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    do_train(args)
