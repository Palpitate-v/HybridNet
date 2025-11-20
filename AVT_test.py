
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_AVT import AVT_dataset
from utils import test_single_volume_AVT
from model.HIFNet import HIFNet

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,Normalize


parser = argparse.ArgumentParser()
draw='false'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser.add_argument('--checkpoint_path', type=str,

default='./checkpoints/demo9.83-3-bs32-dropVit-dropLst0.5-swin_SDI-head488-e128-dim128-ce5_di5-T170_AdamW_cos_0.001_170_AVT/epoch_160.pth')

parser.add_argument('--is_savenii', default=''
                    ,action="store_true", help='whether to save results during inference')

parser.add_argument('--volume_path', type=str,
                    default='./data/AVT/test', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='AVT', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_AVT', help='list dir')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')

parser.add_argument('--test_save_dir', type=str, default='', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--model_name', type=str,
                    default=" ", help='the name of network')
args = parser.parse_args()


def save_figure(fig, filename, folder=args.checkpoint_path.split('.pth')[0]):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath)
    plt.close(fig)

def label2color(label, colormap):
    """将标签图转换为彩色图像"""
    norm = Normalize(vmin=0, vmax=colormap.N)
    return colormap(norm(label))

def plot_images(original, predicted, label, title="Image Comparison"):
    # Convert tensors to numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.squeeze().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.squeeze().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.squeeze().cpu().numpy()

    colors = [(0, 0, 0),  # Background
              (255, 0, 0)]  # Class 1
    colormap = ListedColormap(colors)

    # 如果label是二维的，那么不需要遍历切片
    if label.ndim == 2:
        if np.any(label > 0):
            # Convert label map to color image
            label_colored = label2color(label, colormap)
            predict_colored = label2color(predicted, colormap)

            # Create a figure and a set of subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')

            # Predicted image
            axes[1].imshow(predict_colored, cmap=colormap)
            axes[1].set_title("Predicted")
            axes[1].axis('off')

            # Colored label image
            axes[2].imshow(label_colored, cmap=colormap)
            axes[2].set_title("Label")
            axes[2].axis('off')

            plt.suptitle(f"{title}")
            save_figure(fig, f"{title}.png")

    else:
        # 如果label是三维的，遍历每个切片
        for i in range(label.shape[0]):
            if np.any(label[i, :, :] > 0):
                original_slice = original[i, :, :] if original.ndim == 3 else original
                predicted_slice = predicted[i, :, :] if predicted.ndim == 3 else predicted
                label_slice = label[i, :, :] if label.ndim == 3 else label

                # Convert label map to color image
                label_colored = label2color(label_slice, colormap)
                predict_colored = label2color(predicted_slice, colormap)

                # Create a figure and a set of subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(original_slice, cmap='gray')
                axes[0].set_title("Original")
                axes[0].axis('off')

                # Predicted image
                axes[1].imshow(predict_colored, cmap=colormap)
                axes[1].set_title("Predicted")
                axes[1].axis('off')

                # Colored label image
                axes[2].imshow(label_colored, cmap=colormap)
                axes[2].set_title("Label")
                axes[2].axis('off')

                plt.suptitle(f"{title} - Slice {i}")
                save_figure(fig, f"{title}_slice_{i}.png")



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
        if(draw=='true'):
            plot_images(image, predicted, label, title=case_name)

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
        'AVT': {
            'Dataset': AVT_dataset,
            "checkpoint_path" :args.checkpoint_path,
            'list_dir': './lists/lists_AVT',
            'num_classes': 2,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.checkpoint_path = dataset_config[dataset_name]['checkpoint_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    net = HIFNet(num_classes=args.num_classes).cuda()
    net.load_state_dict(torch.load(args.checkpoint_path))

    log_folder = args.checkpoint_path
    snapshot_name = log_folder[:-4]
    logging.basicConfig(filename=snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        print('------------------------------------save---nii-----------------------------------------')
        args.test_save_dir = snapshot_name + "_pre"
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    if args.dataset == 'AVT':
        inference_AVT(args, net, test_save_path)



