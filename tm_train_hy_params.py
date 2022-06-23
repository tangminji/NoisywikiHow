# our method main.py
# tailor-made regularization
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from hyperopt import Trials
from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe
from hyperopt import space_eval

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

if __name__ == '__main__':
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()

    assert args.dataset == "wiki"

    MAX_EVALS = 20  # TODO 设置轮次
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

    if 'STGN' in args.exp_name:
        # space = {
        #     'forget_times': hp.quniform('forget_times', 3, 10, 1), # 只有30epoch,不用设太大
        #     'ratio_l': hp.uniform('ratio_l', 0, 1.0), #loss vs forget的权重,0~1
        #     'avg_steps': hp.choice('avg_steps', [20]),
        #     'times': hp.choice('times', [0, 10, 20, 30, 40, 50, 60]),
        #     'sigma': hp.choice('sigma', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]),
        # }

        # 第二轮
        # space = {
        #     'forget_times': 5, # 只有30epoch,不用设太大
        #     'ratio_l': 0.5, #loss vs forget的权重,0~1
        #     'avg_steps': hp.choice('avg_steps', [20]),
        #     'times': hp.choice('times', [0, 10, 20, 30, 40, 50, 60]),
        #     'sigma': hp.choice('sigma', [1e-6,5e-6,1e-5,5e-5,1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        # }
        # 第三轮
        space = {
            'forget_times': hp.quniform('forget_times', 2, 5, 1), # 只有10epoch,不用设太大
            # 'ratio_l': hp.uniform('ratio_l', 0, 1.0), #loss vs forget的权重,0~1
            'ratio_l': 0.5,
            'avg_steps': hp.choice('avg_steps', [20]),
            'times': hp.choice('times', [ 10, 20, 30]),
            'sigma': hp.choice('sigma', [5e-3, 1e-2]),
            #hp.choice('sigma', [5e-4, 1e-3, 5e-3, 1e-2]),
        }
    if 'GCE' in args.exp_name:
        space = {
            'q': hp.choice('q', [0.4, 0.7, 1.0]),
        }
    if 'SLN' in args.exp_name:
        space = {
            'sigma': hp.choice('sigma', [0.1,0.2,0.5, 1])
        }
    if 'STGN_GCE' in args.exp_name:
        space = {
            'forget_times': hp.quniform('forget_times', 1, 5, 1), # 只有10epoch,不用设太大
            'ratio_l': 0.5,
            'avg_steps': hp.choice('avg_steps', [20]),
            'times': hp.choice('times', [ 10, 20]),
            'sigma': hp.choice('sigma', [5e-3, 1e-2]),
            'q': hp.choice('q', [0.2, 0.4]),
        }
    if 'GNMO' in args.exp_name:
        space = {
            'sigma': hp.choice('sigma',[1e-3,5e-3,1e-2,5e-2,1e-1])
        }
    if 'GNMP' in args.exp_name:
        space = {
            'sigma': hp.choice('sigma',[1e-3,5e-3,1e-2,5e-2,1e-1])
        }

    best = fmin(fn=main, space=space, algo=tpe.suggest,
                max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(args.seed))

    print(best)
    print(bayes_trials.results)
    #TODO: use only using hp.choice
    #https://github.com/hyperopt/hyperopt/issues/284
    #https://github.com/hyperopt/hyperopt/issues/492
    print(space_eval(space, best))
    best = space_eval(space, best)
    args.log_dir = args.exp_name
    json.dump({"best": best, "trials": bayes_trials.results},
              open(os.path.join(args.log_dir, "hy_best_params.json"), "w+", encoding="utf-8"),
              ensure_ascii=False, cls=NpEncoder)
    #os.remove(args.params_path)
    os.remove(args.out_tmp)
