import argparse
from common.losses import *
import os
# Settings
parser = argparse.ArgumentParser(description='Noisywikihow')

# File and Path
parser.add_argument('--exp_name', default='Noisywikihow', type=str)
parser.add_argument('--sub_script', default='sbatch_wiki_hy_subbasemix0.4bart.sh', type=str)
parser.add_argument('--out_tmp', default='wiki_out_tmp.json', type=str)
parser.add_argument('--params_path', default='wiki_params.json', type=str)
parser.add_argument('--log_dir', default='log/test', type=str)

parser.add_argument('--gpu_id', type=int, default=0, help='index of gpu to use')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--restart', default=True, const=True, action='store_const',
                    help='Erase log and saved checkpoints and restart training')#False
parser.add_argument('--mode', type=str, default='no_GN',
                    choices=['no_GN'])

# Data Setting
# max_length=128, num_class=158
parser.add_argument('--dataset', default='wiki', type=str, help="Model type selected in the list: [wiki]")
parser.add_argument('--max_length', type=int ,default=128, help='max step length after truncation')
parser.add_argument('--num_class', default=158, type=int, help="the number of event intention classes (default: 158)")
# Noise
parser.add_argument('--noise_mode', type=str, default='mix', choices=["neighbor", "tail", "uncommon", "mix","sym","idn"],
                    help='Noise mode in the list: [neighbor, tail, uncommon, mix, sym, idn]')
parser.add_argument('--noise_rate', type=float, default=0.0, help='Noise rate')

# Model Setting
parser.add_argument('--model_type', type=str, default='bart', choices=['bert', 'roberta', 'xlnet', 'albert', 'bart','gpt2','t5'])
# TODO use LSTM
# bilstm 2-hidden-layers, need to adjust lr=5e-4
parser.add_argument('--use_lstm',action='store_true',help='use lstm')


# Train Setting
# batch_size=32 epoch=10 (Note:Co-teaching needs 2 models)
parser.add_argument('--loss', default='CE', type=str, help="loss type")
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training') # batch_size=32
parser.add_argument('--test_batch_size', type=int, default=100, help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--data_path', type=str, default='data/wikihow', help='the data and embedding path')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate') # For BART 3e-5, For lstm 5e-4

# TODO MIXUP
# alpha = [0.1, 0.2, 0.4, 1, 8, 32] lambda~beta(alpha,alpha)
parser.add_argument('--alpha', default=1.0, type=float, help='alpha for MIXUP')

# TODO REWEIGHT (data parameter)
parser.add_argument('--learn_class_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per class')
parser.add_argument('--learn_inst_parameters', default=False, const=True, action='store_const',
                    help='Learn temperature per instance')
parser.add_argument('--skip_clamp_data_param', default=False, const=True, action='store_const',
                    help='Do not clamp data parameters during optimization')
parser.add_argument('--lr_class_param', default=0.1, type=float, help='Learning rate for class parameters')
parser.add_argument('--lr_inst_param', default=0.1, type=float, help='Learning rate for instance parameters')
parser.add_argument('--wd_class_param', default=0.0, type=float, help='Weight decay for class parameters')
parser.add_argument('--wd_inst_param', default=0.0, type=float, help='Weight decay for instance parameters')
parser.add_argument('--init_class_param', default=1.0, type=float, help='Initial value for class parameters')
parser.add_argument('--init_inst_param', default=1.0, type=float, help='Initial value for instance parameters')



# TODO SR (Sparse Regularization)
# tau, p, lamb, rho, freq
parser.add_argument('--tau',default=1,type=float,help='tau for SR')
parser.add_argument('--normp',default=0.1,type=float,help='p for SR in pNorm')
parser.add_argument('--lamb',default=5,type=float,help='lamb for SR')
parser.add_argument('--rho',default=1.02,type=float,help='rho for SR')
parser.add_argument('--freq',default=0,type=int,help='freq in SR (Default:0, unused)')

# TODO CT (Co-teaching)
# CV: epoch200,num_gradual10; try 3,5,8
parser.add_argument('--ct_num_gradual', type = int, default = 5, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
# When forget_rate=None，set forget_rate=noise_rate
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)

# TODO SEAL
parser.add_argument('--round', default=4, type=int, help="epochs in one SEAL iteration")

# TODO CNLCU
parser.add_argument('--co_lambda', type=float, help='sigma^2 or tau_min', default=1e-1)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--time_step', type=int, help='time_step', default=5)

# Show result from old model. Don't train a new model
parser.add_argument('--show_result',action='store_true',help='show test result only')


args = parser.parse_args()

args.data_path = "data/wikihow"

# TODO You can either use online model, or download the model in advance.
# e.g. "bert": "model_download_path/bert-base-uncased"
args.model_path = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "xlnet": "xlnet-base-cased",
    "albert": "albert-base-v2",
    "bart": "facebook/bart-base",
    "gpt2": "gpt2",
    "t5": "t5-base",
}

# TODO lstm, adjust the Optimal parameter
if 'lstm' in args.exp_name or args.use_lstm:
    args.use_lstm = True
    # TODO If you want to change the learning rate of lstm, please omit this line
    args.lr = 5e-4


Wiki_CONFIG = {
    "CE": nn.CrossEntropyLoss(reduction='none'),
}

