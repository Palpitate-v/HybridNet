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
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_AVT import AVT_dataset
from utils import test_single_volume_Synapse
from model.HIFNet import HIFNet
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()

draw = 'false'
datafile_name = 'Synapse'

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


def save_figure(fig, filename, folder=args.test_save_dir):
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
              (0, 0, 255),  # Class 1 蓝色
              (0, 255, 0),  # Class 2 绿色
              (255, 0, 0),  # Class 3 红色
              (0, 255, 255),  # Class 4 青蓝色
              (255, 0, 255),  # Class 5 玫红
              (255, 255, 0),  # Class 6 黄色
              (63, 208, 244),  # Class 7 浅蓝色
              (241, 240, 234)]  # Class 8 灰色
    colormap = ListedColormap(colors)

    # 如果label是二维的，那么不需要遍历切片
    if label.ndim == 2:
        if np.any(label > 0):
            label_colored = label2color(label, colormap)
            predict_colored = label2color(predicted, colormap)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # 显示原图像
            axes[0].imshow(original, cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')

            # 确保 predict_colored 是 3 通道的 RGB 图像
            if predict_colored.shape[-1] == 4:
                predict_colored = predict_colored[:, :, :3]

            axes[1].imshow(predict_colored)
            axes[1].set_title("Predicted")
            axes[1].axis('off')

            plt.suptitle(f"{title}")
            save_figure(fig, f"{title}.png")

    else:
        # 如果label是三维的，遍历每个切片
        for i in range(label.shape[0]):
            if np.any(label[i, :, :] > 0):
                original_slice = original[i, :, :] if original.ndim == 3 else original
                predicted_slice = predicted[i, :, :] if predicted.ndim == 3 else predicted
                label_slice = label[i, :, :] if label.ndim == 3 else label

                label_colored = label2color(label_slice, colormap)
                predict_colored = label2color(predicted_slice, colormap)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                # Original image
                axes[0].imshow(original_slice, cmap='gray')
                axes[0].set_title("Original")
                axes[0].axis('off')

                # 确保 predict_colored 是 3 通道的 RGB 图像
                if predict_colored.shape[-1] == 4:
                    predict_colored = predict_colored[:, :, :3]

                axes[1].imshow(predict_colored)
                axes[1].set_title("Predicted")
                axes[1].axis('off')

                plt.suptitle(f"{title} - Slice {i}")
                save_figure(fig, f"{title}_slice_{i}.png")


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

        unique_values, counts = np.unique(predicted, return_counts=True)
        print(unique_values, counts)
        print("________________________________________________________________")
        if(draw=='true'):
            # if i_batch == 0:  # Only plot the first batch for demonstration
            plot_images(image, predicted, label, title=case_name)

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
    # 初始化计时变量
    total_time = 0.0
    total_slices = 0
    total_volumes = 0
    for i_batch, sampled_batch in enumerate(testloader):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        origin, direction, xyz_thickness = sampled_batch["origin"], sampled_batch["direction"], sampled_batch["xyz_thickness"]
        origin = origin.detach().numpy()[0]
        direction = direction.detach().numpy()[0]
        xyz_thickness = xyz_thickness.detach().numpy()[0]
        # 开始计时
        start_time = time.time()

        metric_i,predicted = test_single_volume_AVT(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, origin=origin, direction=direction, xyz_thickness=xyz_thickness)
        # 结束计时
        end_time = time.time()
        inference_time = end_time - start_time
        # 累计统计数据
        total_time += inference_time
        # 获取当前volume的切片数量
        volume_slices = image.shape[1] if image.ndim > 1 else 1
        total_slices += volume_slices
        total_volumes += 1

        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice : %f mean_hd95 : %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        # if(draw=='true'):
        #     plot_images(image, predicted, label, title=case_name)

        # 计算并记录推理时间指标
    if total_time > 0:
        slices_per_second = total_slices / total_time
        volumes_per_minute = (total_volumes / total_time) * 60

        logging.info('=' * 50)
        logging.info('Inference Time Statistics:')
        logging.info('Total inference time: {:.2f} seconds'.format(total_time))
        logging.info('Processed {} slices in {} volumes'.format(total_slices, total_volumes))
        logging.info('Speed: {:.2f} slices/second'.format(slices_per_second))
        logging.info('Speed: {:.2f} volumes/minute'.format(volumes_per_minute))
        logging.info('=' * 50)

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
