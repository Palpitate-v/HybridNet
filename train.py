import argparse
import torch.backends.cudnn as cudnn
from model.HIFNet import HIFNet
from trainer import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

datafile_name = 'Synapse'


parser = argparse.ArgumentParser()

parser.add_argument('--train_path', type=str,
                    default='data/' + datafile_name + '/train/',
                    help='train root dir for data')  

parser.add_argument('--checkpoint_path', type=str,
                    default='weight', help='weight root dir for data')

parser.add_argument('--dataset', type=str,
                    default=datafile_name, help='experiment_name') 

parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_'+datafile_name, help='list dir')

parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')

parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')  

pre='true'

parser.add_argument('--img_size', type=int,default=224
                    )
parser.add_argument('--op_sc', type=str, default="AdamW_cos"
                    )
parser.add_argument('--T_max', type=int, default=250
                    )
parser.add_argument('--base_lr', type=float, default=0.01
                    )
parser.add_argument('--max_epochs', type=int, default=250
                    )

parser.add_argument('--batch_size', type=int, default=32
                    )

parser.add_argument('--model_name', type=str,
                    default="")

args = parser.parse_args()

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
    dataset_name = args.dataset

    dataset_config = {
        'Synapse': {
            "checkpoint_path" : './checkpoints/seed{}_{}_{}'.format(args.model_name, args.seed, args.dataset),
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'AVT': {
            "checkpoint_path" : './checkpoints/seed{}_{}_{}'.format(args.model_name, args.seed,args.dataset),
            'list_dir': './lists/lists_AVT',
            'num_classes': 2,
        },
    }

    args.checkpoint_path = dataset_config[dataset_name]['checkpoint_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.num_classes = dataset_config[dataset_name]['num_classes']

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    net = HIFNet(num_classes=args.num_classes).cuda()  # 获取网络模型
    if pre == 'true':
        pretrained_weights_path = 'swin_tiny_patch4_window7_224.pth'
        pretrained_weights = torch.load(pretrained_weights_path)
        if not torch.cuda.is_available():
            print("CUDA is not available. Weights will be loaded on CPU.")
        else:
            pretrained_weights = {k: v.cuda() for k, v in pretrained_weights.items() if isinstance(v, torch.Tensor)}
        net.load_state_dict(pretrained_weights, strict=False)
        print("----------------------------- using pre train pth ---------------------------")
        trainer(args, net)
    else:
        trainer(args, net)
