"""
load parameter from json file
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from common.utils import log, set_seed, generate_log_dir
from common.utils import set_args
from cmd_args import args
import json
from tm_main import main


# the params will be record in params.json and filepath
def configure(params):
    """Update args according to params, generate folders, set device and seed
    
    Note we save the params in 'params.json', and vital params appear in path

    Args:
        params (dict): the params from json file

    Returns:
        params (dict): the params
    """
    set_args(args)
    args.exp_name = os.path.join(args.exp_name, f'{ITERATION}')

    if 'REWEIGHT' in args.exp_name:
        param_path = []
        if params.get('lr_class_param'):
            args.learn_class_parameters = True
            args.lr_class_param = params['lr_class_param']
            args.wd_class_param = params.get('wd_class_param',0.0)
            args.init_class_param = params.get('init_class_param',1.0)
            param_path.append(f'class{args.init_class_param}_lr{args.lr_class_param}_wd{args.wd_class_param}')

        if params.get('lr_inst_param'):
            args.learn_inst_parameters = True
            args.lr_inst_param = params['lr_inst_param']
            args.wd_inst_param = params.get('wd_inst_param',0.0)
            args.init_inst_param = params.get('init_inst_param',1.0)
            param_path.append(f'inst{args.init_inst_param}_lr{args.lr_inst_param}_wd{args.wd_inst_param}')
        
        if params.get('skip_clamp_data_param'):
            args.skip_clamp_data_param = params['skip_clamp_data_param']
            param_path.append('skip_clamp')
        param_path = '-'.join(param_path)
        args.exp_name = os.path.join(args.exp_name, param_path)


    if 'CT' in args.exp_name:
        args.forget_rate = params.get('forget_rate')
        args.ct_num_gradual = params['ct_num_gradual']
        args.exponent = params['exponent']
        param_path = f'gradual{args.ct_num_gradual}_exponent{args.exponent}'
        if args.forget_rate:
            param_path += f'_forget{args.forget_rate}'
        args.exp_name = os.path.join(args.exp_name, param_path)

    if 'SEAL' in args.exp_name:
        args.round = params['round']
        args.epoch_round = params.get('epochs', args.epochs)
        args.epochs = args.round * args.epoch_round
        args.exp_name = os.path.join(args.exp_name, f'epoch{args.epoch_round}round{args.round}')

    if 'MIXUP' in args.exp_name:
        args.alpha = params.get('alpha', 1.0)
        args.exp_name = os.path.join(args.exp_name, f'alpha{args.alpha}')

    if 'SR' in args.exp_name:
        # TODO try to run 20 epochs
        args.epochs = 20
        if 'tau' in params:
            args.tau  = params['tau']
        if 'normp' in params:
            args.normp = params['normp']
        if 'lamb' in params:
            args.lamb = params['lamb']
        if 'rho' in params:
            args.rho = params['rho']
        if 'freq' in params:
            args.freq = params['freq']
        args.exp_name = os.path.join(args.exp_name, f'tau{args.tau}_normp{args.normp}_lamb{args.lamb}_rho{args.rho}_freq{args.freq}')

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    args.logpath = args.exp_name + '/' + 'log.txt'

    args.log_dir = os.path.join(os.getcwd(), args.exp_name)
    args.save_dir = os.path.join(args.log_dir, 'weights')

    generate_log_dir(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # save params in params.json
    if not 'base' in args.exp_name:
        with open(os.path.join(args.log_dir, 'params.json'), 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False)

    # should be placed after generate_log_dir()
    log(args.logpath, 'Settings: {}\n'.format(args))
    log(args.logpath, 'Params: {}\n'.format(params))

    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu_id)
    set_seed(args)
    return params
            
if __name__ == '__main__':
    print("load params from : ", args.params_path)
    params = json.load(open(args.params_path, 'r', encoding="utf-8")) if 'base' not in args.exp_name else {}
    global ITERATION
    ITERATION = params['ITERATION']
    assert params is not None
    params = configure(params)
    res = main(args=args, params=params)
    res['iteration'] = ITERATION
    json.dump(res, open(args.out_tmp, "w+", encoding="utf-8"), ensure_ascii=False)