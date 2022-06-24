"""
Set the Optimal parameters for each methods.
This code won't load params from json file, since they use best params.

You can change certain params by passing args in command line.

e.g.
python tm_train_hy_nruns.py --lr 5e-5

"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
from common.utils import log, set_seed, generate_log_dir
from common.utils import set_args
from cmd_args import args
import json
from tm_main import main

# Use params to change the setting in args
def configure(params):
    """Update args according to params and args.exp_name, generate folders, set device and seed
    
    Unlike the sub file, we take advantage of the 'args.exp_name' to determine best setting.
    We don't add the vital params in path.

    If you set 'args.show_result', this code will show the best result and prediction from last trial,
    and won't run a new model.

    Args:
        params (dict): the params from json file

    Returns:
        params (dict): the params, which may turn into the best setting
    """
    
    set_args(args)

    # upgrade epochs to 20
    if '-20epochs' in args.exp_name:
        args.epochs = 20

    if 'REWEIGHT' in args.exp_name:
        if params == {}:
            params['lr_inst_param'] = 0.1 if args.noise_mode in ['idn','neighbor'] and args.noise_rate==0.4 else 0.2
            if 'lstm' in args.exp_name:
                args.epochs = 20
                params['lr_inst_param'] = 5.0

        if params.get('lr_class_param'):
            args.learn_class_parameters = True
            args.lr_class_param = params['lr_class_param']
            args.wd_class_param = params.get('wd_class_param',0.0)
            args.init_class_param = params.get('init_class_param',1.0)

        if params.get('lr_inst_param'):
            args.learn_inst_parameters = True
            args.lr_inst_param = params['lr_inst_param']
            args.wd_inst_param = params.get('wd_inst_param',0.0)
            args.init_inst_param = params.get('init_inst_param',1.0)
        
        if params.get('skip_clamp_data_param'):
            args.skip_clamp_data_param = params['skip_clamp_data_param']
    
    # TODO
    if 'CT' in args.exp_name:
        if params == {}:
            if 'lstm' in args.exp_name:
                params['ct_num_gradual'] = 5
            else:
                params['ct_num_gradual'] = 3 if args.noise_mode in ['sym'] else 8
        args.forget_rate = params.get('forget_rate')
        args.ct_num_gradual = params['ct_num_gradual']
        args.exponent = params.get('exponent', 1) # linear change

    # TODO 10 epochs/round, 4 round
    if 'SEAL' in args.exp_name:
        args.round = params.get('round', 4)
        args.epoch_round = params.get('epochs', 10)
        args.epochs = args.round * args.epoch_round

    # TODO alpha = 1
    if 'MIXUP' in args.exp_name:
        args.alpha = params.get('alpha', 1)

    if 'SR' in args.exp_name:
        if params == {}:
            params = {
                'tau': 0.05,
                'lamb': 0,
                'freq': 0
            }
        # sym, lstm has different Optimal parameters
        if 'SR-20epochs-tau0.5' in args.exp_name or 'lstm' in args.exp_name or 'sym' in args.noise_mode:
            args.epochs = 20
            params['tau'] = 0.5

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

    # TODO Other Methods
    
    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
    
    args.log_dir = os.path.join(os.getcwd(), args.exp_name)
    args.save_dir = os.path.join(args.log_dir, 'weights')

    # if show result, don't renew the folder
    if not args.show_result:
        args.logpath = args.exp_name + '/' + 'log.txt'
        generate_log_dir(args)
    else:
        args.logpath = args.exp_name + '/' + 'data_log.txt'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # should be placed after generate_log_dir()
    log(args.logpath, 'Settings: {}\n'.format(args))
    log(args.logpath, 'Params: {}\n'.format(params))

    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu_id)
    set_seed(args)
    return params

# Those methods have best params in this code
# They won't load params from json file again
def fixed_params(exp_name):
    """We have found best params for certain method, which we use default best params, and not load the json file anymore.

    Args:
        exp_name (str): the param 'args.exp_name', which contains the method name

    Returns:
        bool: True for methods which have best params, False otherwise
    """
    fixed = ['base', 'MIXUP', 'SEAL', 'REWEIGHT', 'CT', 'SR']
    for method in fixed:
        if method in exp_name:
            return True
    return False

if __name__ == '__main__':
    print("load params from : ", args.params_path)
    params = json.load(open(args.params_path, 'r', encoding="utf-8"))['best'] if not fixed_params(args.exp_name) else {}
    assert params is not None
    # '--args.show_result'
    # Set args.show_result=True to show the previous best model results
    # Or it will train a new model and cover the old folder
    params = configure(params)

    main(args=args, params=params)