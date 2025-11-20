import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets.dataset_CRC import CRC_dataset, RandomGenerator_test,RandomGenerator
from utils import test_single_volume_CRC
from model.HIFNet import HIFNet
from sklearn.metrics import confusion_matrix


datafile_name='CRC'

myCheckPoint='/tmp/pycharm_project_803/ParaTransCNN-master/checkpoints/ablation-Unet_AdamW_cos_0.01_500_CRC/epoch_499.pth'

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='data/'+datafile_name+'/test_vol',   #/test_vol_no
                    help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default=datafile_name)
parser.add_argument('--num_classes', type=int,
                    default=2)
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_CRC')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--is_savenii', action="store_true")
parser.add_argument('--test_save_dir', type=str, default='')
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to test_vol')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--checkpoint_path', type=str,
                    default=myCheckPoint
                    )

args = parser.parse_args()

def inference_CRC(args, model, test_save_path=None):
    # 从数据集中加载测试数据
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir,
                           transform=transforms.Compose(
                               [RandomGenerator_test(output_size=[args.img_size, args.img_size])]))

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test_vol iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = []
    total_accuracy=0.0
    total_sensitivity=0.0
    total_specificity=0.0
    total_f1_or_dsc=0.0
    total_miou=0.0
    have_num=0

    with open('./lists/lists_CRC/results.txt', 'a', encoding='utf-8') as file:
        file.write('\n' + args.checkpoint_path + '\n')

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        preds = []
        gts = []

        if (sampled_batch['image'].shape[0] == 1):
            sampled_batch['image'] = sampled_batch['image'][0]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i, prediction = test_single_volume_CRC(
            image, label, model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=1
        )

        gts.append(label.squeeze(1).cpu().detach())
        preds.append(prediction)

        metric_list.append((np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        dice_mean = np.mean(metric_i, axis=0)[0]
        hd95_mean = np.mean(metric_i, axis=0)[1]

        logging.info('idx %d case %s mean_dice : %f mean_hd95 : %f' % (
            i_batch, case_name, dice_mean, hd95_mean
        ))

        with open('./lists/lists_CRC/test_vol.txt', 'r', encoding='utf-8') as file:
            file.seek(0)
            for _ in range(i_batch + 1):
                line = file.readline()
                if not line:
                    break
            line = line.strip()

        # 记录全0的数据
        if dice_mean < 0.4:
            with open('./lists/lists_CRC/results.txt', 'a', encoding='utf-8') as file:
                file.write(line + f' dice: {dice_mean:.4f}' + ' , ' + f' hd95: {hd95_mean:.4f}' + '\n')

        confusion=0
        if i_batch % 1 == 0:
            preds = np.array(preds).reshape(-1)
            gts = [g.view(-1) for g in gts]
            gts = torch.cat(gts)

            y_pre = np.where(preds >= 0.5, 1, 0)
            y_true = np.where(gts >= 0.5, 1, 0)

            unique_labels = np.unique(np.concatenate((y_pre, y_true)))

            if len(unique_labels) > 1:
                confusion = confusion_matrix(y_true, y_pre)
                TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
                accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
                sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
                specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
                f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
                miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

                total_accuracy += accuracy
                total_sensitivity += sensitivity
                total_specificity += specificity
                total_f1_or_dsc += f1_or_dsc
                total_miou += miou
                have_num+=1

            else:
                # 如果只有一个类别，则跳过混淆矩阵的计算
                logging.info(
                    "Only one class found in predictions and true labels, skipping confusion matrix calculation.")
                continue


    total_dice_scores = [item[0] for item in metric_list]
    total_hd95_scores = [item[1] for item in metric_list]

    mean_dice = np.mean(total_dice_scores)
    mean_hd95 = np.mean(total_hd95_scores)
    # metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice : %f mean_hd95 : %f' % (
            # i, metric_list[i - 1][0], metric_list[i - 1][1]
            i,mean_dice,mean_hd95
        ))

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (
        mean_dice, mean_hd95
    ))
    log_info = (f'val epoch: {len(db_test)}, '
                f'loss: {np.mean(metric_list):.4f}, '
                f'miou: {round(total_miou / have_num, 4)}, '
                f'f1_or_dsc: {round(total_f1_or_dsc / have_num, 4)}, '
                f'accuracy: {round(total_accuracy /have_num, 4)}, '
                f'specificity: {round(total_specificity / have_num, 4)}, '
                f'sensitivity: {round(total_sensitivity / have_num, 4)} !!!'
                )
    logging.info(log_info)
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
        datafile_name: {
            'Dataset': CRC_dataset,
            'list_dir': './lists/lists_CRC',
            'num_classes': 2,
        }
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']

    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    net = HIFNet(num_classes=args.num_classes).cuda()
    net.load_state_dict(torch.load(args.checkpoint_path))  # 创建 ParaTransCNN 模型实例，并加载相应的权重。

    log_folder = args.checkpoint_path
    snapshot_name = log_folder[:-4]
    logging.basicConfig(filename=snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:  # 是否保存推理后的 .nii 文件。
        args.test_save_dir = snapshot_name + "_pre"  # 日志名+_pre 命名
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)  # 创建目录以保存输出。
        print("save-path:")
        print(test_save_path)
    else:
        test_save_path = None

    if args.dataset == datafile_name:
        inference_CRC(args, net, test_save_path)
