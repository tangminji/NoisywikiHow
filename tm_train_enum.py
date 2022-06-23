# coding=utf-8
"""Enum params for methods"""

import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np


from cmd_args import args

best_acc = 0
ITERATION = 0

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def main(params):
    global ITERATION
    ITERATION += 1
    params['ITERATION'] = ITERATION
    json.dump(params, open(args.params_path, 'w+', encoding="utf-8"), ensure_ascii=False)
    sig = os.system("sh %s" % args.sub_script)
    assert sig == 0
    res = json.load(open(args.out_tmp, 'r', encoding="utf-8"))
    return res

def get_trials(fixed, space, MAX_EVALS):
    for k in space:
        times = len(space[k])
        break
    if times > MAX_EVALS:
        times = MAX_EVALS
    for t in range(times):
        params = {k: space[k][t] for k in space}
        params.update(fixed)
        yield params

# TODO 每个实验传一个字典，可以让各次实验调参数不完全一样
def get_trials_list(fixed, space, MAX_EVALS):
    times = len(space) 
    if times > MAX_EVALS:
        times = MAX_EVALS
    for t in range(times):
        params = space[t]
        params.update(fixed)
        yield params


# 自己设定要枚举的实验，每个实验只做一次

if __name__ == '__main__':

    assert args.dataset == "wiki"

    MAX_EVALS = 10  # TODO 设置轮次

    fixed = {

    }
    space = {

    }
    # TODO: times, sigma (key hyperparameters)
    # space中所有参数都需要是hp对象，否则best会缺失相应超参数值
    # --noise_rate 0.4 \ #generally known
    # --forget_times 10 \ discrete value
    # --ratio_l 1.0 \
    # --times 50.0 \
    # --avg_steps 20 \
    # --sigma 1e-3 \
    # --sig_max  2e-3 \ #can be obtained by sigma
    # --lr_sig 1e-4 \ #can be obtained by sigma

    # !-- OLD Trial
    if 'STGN' in args.exp_name:
        # 调sigma best sigma 1e-2(数据上选择了1e-3)
        # times=20比times>20好
        # fixed = {
        #     'forget_times': 3,
        #     'ratio_l': 0.5,
        #     'avg_steps': 20,
        #     'times': 30,
        # }
        # space = {
        #     'sigma': [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        # }

        # 调 sigma, times
        # 最优 sigma=0.01, times=20
        # fixed = {
        #     'forget_times': 3,
        #     'ratio_l': 0.5,
        #     'avg_steps': 20,
        # }
        # space = {
        #     'sigma': [],
        #     'times': []
        # }
        # for sigma in [1e-3,5e-3,1e-2]:
        #     for times in [10, 20]:
        #         space['sigma'].append(sigma)
        #         space['times'].append(times)

        # 调 forget_times
        fixed = {
            'ratio_l': 0.5,
            'avg_steps': 20,
            'sigma': 0.01,
            'times': 20,
        }
        space = {
            'forget_times': [1, 2, 3, 4],
        }
    if 'GCE' in args.exp_name:
        # space = {
        #     'q': [0.1, 0.4, 0.7, 1.0],
        # }
        space = {
            'q': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        }
    
    '''
    {'sigma': 0.05, 'ITERATION': 1} -53.13291139240506
    [{'loss': -53.13291139240506, 'test_at_best_top1': 53.13291139240506, 'test_at_best_top5': 82.29746835443038, 'stable_acc_top1': 53.03417721518987, 'stable_acc_top5': 82.29746835443038, 'params': {'sigma': 0.05, 'ITERATION': 1}, 'train_time': 8989.119092464447, 'status': 'ok', 'iteration': 1}, {'loss': -51.70886075949367, 'test_at_best_top1': 51.70886075949367, 'test_at_best_top5': 81.37974683544304, 'stable_acc_top1': 51.21139240506329, 'stable_acc_top5': 80.95443037974682, 'params': {'sigma': 0.1, 'ITERATION': 2}, 'train_time': 8883.598945140839, 'status': 'ok', 'iteration': 2}, {'loss': -47.348101265822784, 'test_at_best_top1': 47.348101265822784, 'test_at_best_top5': 78.29746835443038, 'stable_acc_top1': 45.51139240506329, 'stable_acc_top5': 77.30379746835442, 'params': {'sigma': 0.2, 'ITERATION': 3}, 'train_time': 9043.852854251862, 'status': 'ok', 'iteration': 3}]
    sigma = 0.05时峰值最好，到sigma=0.2时峰值显著下降
    不过stable_acc没看到
    '''

    if 'SLN' in args.exp_name:
        space = {
            'sigma': [0.05, 0.1, 0.2] #0.01效果太不显著，和基线效果差不多
            # 'sigma': [0.01, 0.05, 0.1, 0.2]
            # 'sigma': [0.1,0.2,0.5, 1]
        }
    
    
    if 'STGN_GCE' in args.exp_name:
        space = {
            'sigma': [],
            'times': [],
            'q': []
        }
        fixed = {
            'ratio_l': 0.5,
            'avg_steps': 20,
            # 'sigma': 0.01,
            # 'times': 20,
            'forget_times': 3
        }
        for q in [0.2,0.4]:
            for sigma in [0.005, 0.01]:
                for times in [10, 20]:
                    space['sigma'].append(sigma)
                    space['times'].append(times)
                    space['q'].append(q)
    # if 'STGN_GCE' in args.exp_name:
    #     space = {
    #         'forget_times': hp.quniform('forget_times', 3, 8, 1), # 只有30epoch,不用设太大
    #         'ratio_l': hp.uniform('ratio_l', 0, 1.0), #loss vs forget的权重,0~1
    #         'avg_steps': hp.choice('avg_steps', [20]),
    #         'times': hp.choice('times', [ 20, 30, 40, 50, 60]),
    #         'sigma': hp.choice('sigma', [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
    #         'q': hp.choice('q', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5, 0.7]),
    #     }
    if 'GNMO' in args.exp_name:
        space = {
            'sigma': [1e-3,5e-3,1e-2,5e-2,1e-1]
        }
    if 'GNMP' in args.exp_name:
        space = {
            'sigma': [1e-3,5e-3,1e-2,5e-2,1e-1]
        }
    # --! OLD Trial

    # TODO 添加新方法参数
    if 'LOSS' in args.exp_name:
        # NCEandAGCE(alpha=0, beta=1, num_classes=10, a=4, q=0.2), MNIST
        # NCEandAGCE(alpha=1, beta=4, num_classes=10, a=6, q=1.5), CIFAR-10
        # NCEandAGCE(alpha=10, beta=0.1, num_classes=100, a=1.8, q=3), CIFAR-100
        # 
        # 用MNIST的参数根本train不起来

        space = {
            'a':     [1.8,
              #4,    6,      2.5
              ],
            'q':     [3,
                # 0.2,  1.5,    3
                ],
            'alpha': [10,
            #    0,    1,   50
               ],
            'beta':  [0.1,
            #   1,    4,    0.1
              ]
        }
    '''
    python main_cifar.py \
    --rand_fraction 0.4 \
    --init_inst_param 1.0 \
    --lr_inst_param 0.2 \
    --wd_inst_param 0.0 \
    --learn_inst_parameters 

    python main_imagenet.py \
  --arch 'resnet18' \ 
  --data 'path/to/imagenet' \
  --init_class_param 1.0 \
  --lr_class_param 0.1 \
  --wd_class_param 1e-4 \
  --init_inst_param 0.001 \
  --lr_inst_param 0.8 \
  --wd_inst_param 1e-8 \
  --learn_class_paramters \
  --learn_inst_parameters 
    
    文中建议 init_class_param + init_inst_param 接近1.0
    param = class_param + inst_param
    param相当于温度tau： param=1时是CE， param<1时更重视最大类别，param>1时会忽略类别概率
    噪声样本 param 上升， 干净样本 param 下降
    '''
    # old REWEIGHT
    if 'REWEIGHT-1' in args.exp_name:
        # space = {
        #     'init_inst_param':  [1.0,   0.001],
        #     'lr_inst_param':    [0.2,   0.8],
        #     'wd_inst_param':    [0.0,   1e-8],
        #     'init_class_param': [None,  1.0],
        #     'lr_class_param':   [None,  0.1],
        #     'wd_class_param':   [None,  1e-4],
        # }
        fixed = {
            'init_inst_param': 1.0,
            'wd_inst_param': 0.0, # 由于区分度不够大，所以不设置weight_decay
            'init_class_param': None,
            'lr_class_param': None,
            'wd_class_param': None,
        }
        space = {
            'lr_inst_param':    [0.1, 0.5, 0.8, 1.0, 2.0, 5.0], # 调整学习率，提高tau的区分度
        }
    if 'REWEIGHT-2' in args.exp_name:
        # space = {
        #     'init_inst_param':  [1.0,   0.001],
        #     'lr_inst_param':    [0.2,   0.8],
        #     'wd_inst_param':    [0.0,   1e-8],
        #     'init_class_param': [None,  1.0],
        #     'lr_class_param':   [None,  0.1],
        #     'wd_class_param':   [None,  1e-4],
        # }
        fixed = {
            'init_inst_param': 1.0,
            'wd_inst_param': 0.0, # 由于区分度不够大，所以不设置weight_decay
            'init_class_param': None,
            'lr_class_param': None,
            'wd_class_param': None,
        }
        space = {
            'lr_inst_param':    [10, 15, 20, 25, 50], # 调整学习率，提高tau的区分度
        }
    # 看上去REWEIGHT的学习率还能继续增大
    # 目前调整forget_rate = 1.25 * noise_rate (实测不如noise_rate)
    # 个人感觉ct_num_gradual设置为3更合适，后期有更多稳定时间；但是如果能1个epoch内多次渐变就更好了
    if 'CT' in args.exp_name:
        fixed = {
            # 'forget_rate': None,
            'exponent': 1,
        }
        space = {
            'ct_num_gradual': [3,5,8]
        }
    if 'SEAL' in args.exp_name:
        space = {
            'round': [2,4,8]
        }

        # 先减少seal_t，发现问题
    if 'MIXUP' in args.exp_name:
        space = {
            'alpha': [0.2, 0.4, 1, 8]
        }
    if 'SR' in args.exp_name:
        #tau, p, lamb, rho, freq
        # CIFAR-100, CIFAR-10, MNIST的参数
        # 对我们数据集不合适，模型、数据、epoch数（10vs150~200）都不相同
        space = {
            'tau':  [0.5,   0.5,    0.1],
            'normp':[0.01,  0.1,    0.1],
            'lamb': [10,    1.2,    4],
            'rho':  [1.02,  1.03,   2],
            'freq': [1,     1,      5],
        }
    # SR tau 调参 0.01, 0.1, 0.3, 0.5, 0.7, 1.0
    # lamb 调参 0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 10, 15, 20
    # p 0.01, 0.1, 0.3, 0.5, 0.7, 1.0
    # 调参时默认lamb固定
    # 最好保证lamb0*p<1, lamb大时可后期缓慢增加
    # 最优超参  tau=0.05
    if 'SR-tau-1' in args.exp_name:
        space = {
            'tau': [0.01, 0.05, 0.1]
        }
        fixed = {
            'lamb': 0,
        }
    if 'SR-tau-2' in args.exp_name:
        space = {
            'tau': [0.3, 0.5, 0.7, 1.0]
        }
        fixed = {
            'lamb': 0,
        }
    if 'SR-lamb' in args.exp_name:
        space = {
            'lamb': [0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 10, 15, 20]
        }
    if 'SR-lamb-p0.7-1' in args.exp_name:
        fixed = {
            'normp': 0.7
        }
        space = {
            'lamb': [0.1, 0.3, 1.0, 3.0]
        }
    if 'SR-lamb-p0.7-2' in args.exp_name:
        fixed = {
            'normp': 0.7
        }
        space = {
            'lamb': [5.0, 7.0, 10, 15, 20]
        }
    if 'SR-lamb-p0.9-1' in args.exp_name:
        fixed = {
            'normp': 0.9
        }
        space = {
            'lamb': [0.1, 0.3, 1.0, 3.0]
        }
    if 'SR-lamb-p0.9-2' in args.exp_name:
        fixed = {
            'normp': 0.9
        }
        space = {
            'lamb': [5.0, 7.0, 10, 15, 20]
        }
    
    # tau=0.05, p=0.7 or 0.9 调lamb
    if 'SR-lamb-tau0.05-p0.7-1' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'normp': 0.7
        }
        space = {
            'lamb': [0.1, 0.3, 1.0, 3.0]
        }
    if 'SR-lamb-tau0.05-p0.7-2' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'normp': 0.7
        }
        space = {
            'lamb': [5.0, 7.0, 10, 15, 20]
        }
    if 'SR-lamb-tau0.05-p0.9-1' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'normp': 0.9
        }
        space = {
            'lamb': [0.1, 0.3, 1.0, 3.0]
        }
    if 'SR-lamb-tau0.05-p0.9-2' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'normp': 0.9
        }
        space = {
            'lamb': [5.0, 7.0, 10, 15, 20]
        }
    # if 'SR-p' in args.exp_name:
    #     space = {
    #         'normp': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0]
    #     }

    # 对lstm，最优超参tau0.5

    if 'SR-p-1' in args.exp_name:
        fixed = {
            'tau': 0.5,
        }
        space = {
            'normp': [0.01, 0.05]
            # 'normp': [0.01, 0.05, 0.1]
        }
    if 'SR-p-2' in args.exp_name:
        fixed = {
            'tau': 0.5
        }
        space = {
            'normp': [0.1, 0.3]
            # 'normp': [0.3, 0.5, 0.7, 1.0]
        }
    if 'SR-p-3' in args.exp_name:
        fixed = {
            'tau': 0.5
        }
        space = {
            'normp': [0.5, 0.7]
        }
    if 'SR-p-4' in args.exp_name:
        fixed = {
            'tau': 0.5
        }
        space = {
            'normp': [0.9, 1.0]
        }

    # SR 组合
    if 'SR-new' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'normp': 0.01
        }
        space = [
            {'lamb': 0.1},
            {'lamb': 1.0},
            {'lamb': 0.1, 'rho': 1.5, 'freq': 1}, # 缓慢增长50倍
            {'lamb': 1.0, 'rho': 1.5, 'freq': 1}
        ]
    # TODO 20epoch的话，rho=1.22
    # 这里尝试使用CIFAR100的参数，更改rho
    # 就效果来看，cifrar100的超参数非常不适合本数据集
    if 'SR-CF' in args.exp_name:
        fixed = {
            'tau': 0.5,
            'normp': 0.01,
            'lamb': 10,
            'freq': 1
        }
        space = [
            {'rho': 1.22},
            {'rho': 1.5},
        ]
    # TODO tau=0.05 改变p, rho
    if 'SR-rho-tau0.05' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'normp': 0.7,
            'lamb': 0.3,
            'freq': 1
        }
        space = [
            {'rho': 1.22},
            {'rho': 1.5},
        ]
    # 看上去然后rho1.22的效果还可以，先调一下
    if 'SR-rho1.22-tau0.05-p-1' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'rho': 1.22,
            'lamb': 0.3,
            'freq': 1
        }
        space = {
            'normp': [0.01, 0.05, 0.1]
        }
    if 'SR-rho1.22-tau0.05-p-2' in args.exp_name:
        fixed = {
            'tau': 0.05,
            'rho': 1.22,
            'lamb': 0.3,
            'freq': 1
        }
        space = {
            'normp': [0.3, 0.5, 0.9, 1.0]
        }
    # tau=0.5 p=0.5 调lamb; tau=0.5效果最好(57.05)
    if 'SR-rho1.22-tau0.5-p0.5-lamb-1' in args.exp_name:
        fixed = {
            'tau': 0.5,
            'rho': 1.22,
            'freq': 1,
            'normp': 0.5
        }
        space = {
            'lamb': [0.1, 0.3, 0.5, 1.0, 3.0]
        }
    if 'SR-rho1.22-tau0.5-p0.5-lamb-2' in args.exp_name:
        fixed = {
            'tau': 0.5,
            'rho': 1.22,
            'freq': 1,
            'normp': 0.5
        }
        space = {
            'lamb': [5.0, 7.0, 10, 15, 20]
        }

    # tau=0.5 p=0.7 调lamb; tau=0.5效果最好(57.05)
    if 'SR-rho1.22-tau0.5-p0.7-lamb-1' in args.exp_name:
        fixed = {
            'tau': 0.5,
            'rho': 1.22,
            'freq': 1,
            'normp': 0.7
        }
        space = {
            'lamb': [0.1, 0.3, 0.5, 1.0, 3.0]
        }
    if 'SR-rho1.22-tau0.5-p0.7-lamb-2' in args.exp_name:
        fixed = {
            'tau': 0.5,
            'rho': 1.22,
            'freq': 1,
            'normp': 0.7
        }
        space = {
            'lamb': [5.0, 7.0, 10, 15, 20]
        }


    if type(space)==list:
        trials = get_trials_list(fixed, space, MAX_EVALS)
    else:
        trials = get_trials(fixed, space, MAX_EVALS)
    all_trials = []
    best_loss = None
    best = None
    for params in trials:
        res = main(params)
        all_trials.append(res)
        loss = res['loss']
        if best_loss is None or loss<best_loss:
            best_loss = loss
            best = params

    print(best, best_loss)
    print(all_trials)
    #TODO: use only using hp.choice
    #https://github.com/hyperopt/hyperopt/issues/284
    #https://github.com/hyperopt/hyperopt/issues/492
    # print(space_eval(space, best))
    # best = space_eval(space, best)
    args.log_dir = args.exp_name
    json.dump({"best": best, "trials": all_trials},
              open(os.path.join(args.log_dir, "hy_best_params.json"), "w+", encoding="utf-8"),
              ensure_ascii=False, cls=NpEncoder)
    #os.remove(args.params_path)
    os.remove(args.out_tmp)
