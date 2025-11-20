import functools
import logging
import os
import random
import sys
import time
from datetime import datetime

from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets.dataset_synapse import *

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, test_single_volume_CRC
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math


datafile_name='AVT'

def worker_init_fn(worker_id, seed):
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def cosine_annealing_lr(T_max, eta_min, eta_max):
    def scheduler(epoch):
        return eta_min + (eta_max - eta_min) * (0.5 - 0.5 * math.cos(math.pi * epoch / T_max))

    return scheduler

def trainer(args, model):
    current_date = datetime.now().strftime('%Y-%m-%d')
    # 记录日志
    logging.basicConfig(filename=args.checkpoint_path + "/log_" + current_date + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    if args.dataset == datafile_name:  # 默认
        db_train = Synapse_dataset(base_dir=args.train_path, list_dir='./lists/lists_AVT', split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
        db_validate = Synapse_dataset(base_dir='data/'+datafile_name+'/test_vol/', list_dir='./lists/lists_AVT', split="test_vol",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(
        db_train,
        # batch_size=args.batch_size,
        batch_size=batch_size,  # !!!
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=functools.partial(worker_init_fn, seed=args.seed)
    )
    val_loader = DataLoader(db_validate,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8,
                            drop_last=True)

    if args.n_gpu > 1:
        print("多GPU并行处理ing...")
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-08,lr=args.base_lr)
    scheduler=CosineAnnealingLR(optimizer,T_max=args.T_max,eta_min = 0.00001) #..............................

    writer = SummaryWriter(args.checkpoint_path + '/log')
    iter_num = 0
    lst_iter_num=0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    start_time = time.time()
    all_epoch_time = max_iterations / 3600

    epoch_train_losses = []
    for epoch_num in iterator:

        total_loss=0.0
        total_loss_ce=0.0
        total_loss_dice=0.0
        lst_iter_num=iter_num

        current_lrs = [group['lr'] for group in optimizer.param_groups]
        # writer.add_scalar('info/lr', current_lrs[0], iter_num)  # 记录到TensorBoard
        lr_=current_lrs[0]

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda().repeat(1, 3, 1, 1), label_batch.cuda()

            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)  # 转为0 和 1了
            loss = 0.5 * loss_ce + 0.5* loss_dice

            total_loss+=loss
            total_loss_ce+=loss_ce
            total_loss_dice+=loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            current_time = datetime.now()
            current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            logging.info(
                'time: %s,time(h): (%f/%f), epoch: (%d/%d), iteration: %d, loss: %f, loss_ce: %f, loss_dice: %f, lr: %f' % (
                current_time_str, ((time.time() - start_time) * iter_num) / 3600,
                (time.time() - start_time) * all_epoch_time, epoch_num, max_epoch, iter_num, loss.item(),
                loss_ce.item(), loss_dice.item(), lr_))
            start_time = time.time()


            if iter_num % 20 == 0:
                if image_batch.size(0) > 1:
                    image = image_batch[1, 0:1, :, :]
                else:
                    print("Batch size is not sufficient.")
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)


        rounded_total_loss = torch.round(total_loss / (iter_num - lst_iter_num), decimals=4)
        rounded_total_loss_ce = torch.round(total_loss_ce / (iter_num - lst_iter_num), decimals=4)
        rounded_total_loss_dice = torch.round(total_loss_dice / (iter_num - lst_iter_num), decimals=4)

        #每一次迭代，记录总loss
        logging.info("epoch:(%d / %d), now_loss:%f , now_loss_ce:%f , now_loss_dice:%f .",
                     epoch_num,max_epoch,
                     rounded_total_loss.item(),
                     rounded_total_loss_ce.item(),
                     rounded_total_loss_dice.item())

        epoch_train_losses.append(rounded_total_loss.item())

        save_mode_path = os.path.join(args.checkpoint_path, 'last_train.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

        if epoch_num>=150 and epoch_num<=200 and epoch_num%10==0:
            save_mode_path = os.path.join(args.checkpoint_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        if epoch_num>200 and epoch_num%10==0:
            save_mode_path = os.path.join(args.checkpoint_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 5:
            save_mode_path = os.path.join(args.checkpoint_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        scheduler.step()
        writer.add_scalar('info/lr_epoch_end', optimizer.param_groups[0]['lr'], epoch_num)


    print(epoch_train_losses)
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_train_losses, label='Training Loss')
    plt.title(args.checkpoint_path)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # 保存图像
    plt.savefig(args.checkpoint_path + '/training_loss.png')
    plt.show()
    plt.close()

    writer.close()
    return "Training Finished!"



