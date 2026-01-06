import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_AVT import AVT_dataset
from utils import test_single_volume_Synapse, test_single_volume_AVT
from model.HIFNet import HIFNet
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()

draw = 'false'
datafile_name = 'Synapse'

parser.add_argument('--checkpoint_path', type=str, default='',help='Set the path to the trained weights here.' )
parser.add_argument('--volume_path', type=str,
                    default='./data/'+datafile_name+'/test_vol',
                    help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default=datafile_name, help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_'+datafile_name, help='list dir')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='predict/', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size per gpu')
parser.add_argument('--max_epochs', type=int,
                    default=250, help='maximum epoch number to train')
parser.add_argument('--dim', type=int,
                    default=64, help='')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--model_name', type=str,
                    default="", help='the name of network')
args = parser.parse_args()


def inference_Synapse(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test_vol iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i, predicted = test_single_volume_Synapse(image, label, model, classes=args.num_classes,
                                                 patch_size=[args.img_size, args.img_size],
                                                 test_save_path=test_save_path, case=case_name, z_spacing=1)

        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice : %f mean_hd95 : %f' % (
        i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice : %f mean_hd95 : %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


def inference_AVT(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        origin, direction, xyz_thickness = sampled_batch["origin"], sampled_batch["direction"], sampled_batch["xyz_thickness"]
        origin = origin.detach().numpy()[0]
        direction = direction.detach().numpy()[0]
        xyz_thickness = xyz_thickness.detach().numpy()[0]

        metric_i,predicted = test_single_volume_AVT(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, origin=origin, direction=direction, xyz_thickness=xyz_thickness)

        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice : %f mean_hd95 : %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            "checkpoint_path": args.checkpoint_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'dim':64,
        },
        'AVT': {
            'Dataset': AVT_dataset,
            "checkpoint_path":args.checkpoint_path,
            'list_dir': './lists/lists_AVT',
            'num_classes': 2,
            'dim':128,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.checkpoint_path = dataset_config[dataset_name]['checkpoint_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.dim = dataset_config[dataset_name]['dim']

    net = HIFNet(num_classes=args.num_classes,dim=args.dim).cuda()
    net.load_state_dict(torch.load(args.checkpoint_path))

    log_folder = args.checkpoint_path
    snapshot_name = log_folder[:-4]
    logging.basicConfig(filename=snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = snapshot_name + "_pre"
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    if args.dataset == 'Synapse':
        inference_Synapse(args, net, test_save_path)
    else:
        inference_AVT(args, net, test_save_path)
