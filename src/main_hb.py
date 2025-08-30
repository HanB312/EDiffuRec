import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model_hb import create_model_diffu, Att_Diffuse_model
from trainer import model_train, LSHT_inference
from collections import Counter

from thop import profile, clever_format
from torchvision.models import *
from torchsummary import summary
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon_beauty', help='Dataset name: toys, amazon_beauty, steam, ml-1m')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=25, help='Random seed') # 1997 # 25
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks') #4
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')  ## 500
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR') #100
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR') #0.1
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam','AdamW']) # + Adamw
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') # 0.001
parser.add_argument('--loss_lambda', type=float, default=0.001, help='loss weight for diffusion') # 0.001
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization') #0
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step') #32
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight') # 0.001
parser.add_argument('--noise_schedule', default='trunc_lin',
                    help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=20, help='the number of epoch to eval') #20
parser.add_argument('--patience', type=int, default=1, help='the number of epoch to wait before early stop') #5
parser.add_argument('--description', type=str, default='Diffu_norm_score', help='Model brief introduction')
parser.add_argument('--diversity_measure', default=False, help='Measure the diversity of recommendation results')
parser.add_argument('--epoch_time_avg', default=False, help='Calculate the average time of one epoch training')

#warmup
parser.add_argument('--duration', type=int, default=20, help='warmup duration')
parser.add_argument('--duration_num', type=str, default='20', help='warmup duration num str')

## params for noise distribution
parser.add_argument('--noise_dist', type=str, default='weibull', help="type of noise distribution")
# gaussian, gamma, logNormal, gaussian_log, gaussian_beta, gaussian_2
parser.add_argument('--g_alpha', type=float, default=2., help='gamma distribution parameter : alpha')
parser.add_argument('--g_beta', type=float, default=1.7, help='gamma distribution parameter : beta')
parser.add_argument('--log_mean', type=float, default=0, help='logNormal distribution parameter : mean')
parser.add_argument('--log_std', type=float, default=0.5, help='logNormal distribution parameter : std')
parser.add_argument('--w_mean', type=float, default=2, help='logNormal distribution parameter : mean')
parser.add_argument('--w_std', type=float, default=0.5, help='logNormal distribution parameter : std')
parser.add_argument('--a', type=float, default=0.5)
args = parser.parse_args()

print(args)

if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset)

logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S",
                                                                                                    time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num
    return args



def main(args):
    fix_random_seed_as(args.random_seed)
    path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    # cold_hot_long_short(data_raw, args.dataset)

    args = item_num_create(args, len(data_raw['smap']))
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()

    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)


    print('papram',sum([param.nelement() for param in rec_diffu_joint_model.parameters()]))

    #flops, params = profile(rec_diffu_joint_model,inputs=(torch.zeros(12102, 128),), verbose=False)
    #print(flops,params)

    best_model, test_results = model_train(tra_data_loader, val_data_loader, test_data_loader, rec_diffu_joint_model,
                                           args, logger)



if __name__ == '__main__':
    main(args)
