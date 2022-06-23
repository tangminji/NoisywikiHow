import argparse


# 快速创建脚本

results_shell='''#!/bin/bash
            
#SBATCH -J {method}_{noise_mode}_results
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 1-00:00:00
#SBATCH --mem 20240
#SBATCH -e output/{method}_{noise_mode}_results.err
#SBATCH -o output/{method}_{noise_mode}_results.txt

source ~/.bashrc
conda activate base

noise_mode={noise_mode}
method={method}
model_type={model_type}
i=0

for noise_rate in 0.1 0.2
do
python tm_train_hy_nruns.py \\
  --dataset wiki \\
  --show_result \\
  --noise_mode $noise_mode \\
  --noise_rate $noise_rate \\
  --model_type $model_type \\
  --seed $i \\
  --exp_name ../Noisywikihow_output/$noise_mode/nrun/$model_type/wiki_$method/nr$noise_rate/seed$i \\
  --params_path best_params$method$noise_mode$noise_rate$model_type.json \\
  --out_tmp wiki_out_tmp$method$noise_mode$noise_rate$model_type.json \\
  --sub_script sbatch_wiki_hy_sub$method$noise_mode$noise_rate$model_type.sh
done
'''


nrun_shell='''#!/bin/bash
            
#SBATCH -J {method}_nrun{noise_rate}_{noise_mode}_{model_type}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100s-pcie-32gb:1
#SBATCH -t 1-00:00:00
#SBATCH --mem 20240
#SBATCH -e output/{method}_nrun{noise_rate}_{noise_mode}_{model_type}.err
#SBATCH -o output/{method}_nrun{noise_rate}_{noise_mode}_{model_type}.txt

source ~/.bashrc
conda activate base

noise_mode={noise_mode}
noise_rate={noise_rate}
method={method}
model_type={model_type}

for i in 0
do
python tm_train_hy_nruns.py \\
  --dataset wiki \\
  --noise_mode $noise_mode \\
  --noise_rate $noise_rate \\
  --model_type $model_type \\
  --seed $i \\
  --exp_name ../Noisywikihow_output/$noise_mode/nrun/$model_type/wiki_$method/nr$noise_rate/seed$i \\
  --params_path best_params$method$noise_mode$noise_rate$model_type.json \\
  --out_tmp wiki_out_tmp$method$noise_mode$noise_rate$model_type.json \\
  --sub_script sbatch_wiki_hy_sub$method$noise_mode$noise_rate$model_type.sh
done
'''

hy_shell='''#!/bin/bash
            
#SBATCH -J {method}_hy_{noise_rate}_{noise_mode}_{model_type}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 1-00:00:00
#SBATCH --mem 20240
#SBATCH -o output/{method}_hy{noise_rate}_{noise_mode}_{model_type}.out
#SBATCH -e output/{method}_hy{noise_rate}_{noise_mode}_{model_type}.err


source ~/.bashrc
conda activate base

i=0
noise_mode={noise_mode}
noise_rate={noise_rate}
method={method}
model_type={model_type}

python tm_train_hy_params.py \\
--dataset wiki \\
--noise_mode $noise_mode \\
--noise_rate $noise_rate \\
--model_type $model_type \\
--seed $i \\
--exp_name ../Noisywikihow_output/$noise_mode/hy/$model_type/wiki_$method/nr$noise_rate/ \\
--params_path wiki_params$method$noise_mode$noise_rate$model_type.json \\
--out_tmp wiki_out_tmp$method$noise_mode$noise_rate$model_type.json \\
--sub_script sbatch_wiki_hy_sub$method$noise_mode$noise_rate$model_type.sh
'''

sub_shell='''#!/bin/bash

i=0
noise_mode={noise_mode}
noise_rate={noise_rate}
method={method}
model_type={model_type}

python tm_train_hy_sub.py \\
--dataset wiki \\
--noise_mode $noise_mode \\
--noise_rate $noise_rate \\
--model_type $model_type \\
--seed $i \\
--exp_name ../Noisywikihow_output/$noise_mode/hy/$model_type/wiki_$method/nr$noise_rate/ \\
--params_path wiki_params$method$noise_mode$noise_rate$model_type.json \\
--out_tmp wiki_out_tmp$method$noise_mode$noise_rate$model_type.json \\
--sub_script sbatch_wiki_hy_sub$method$noise_mode$noise_rate$model_type.sh
'''

# 枚举次数可能很多，干脆设成2天
enum_shell='''#!/bin/bash
            
#SBATCH -J {method}_em_{noise_rate}_{noise_mode}_{model_type}
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:geforce_rtx_2080_ti:1
#SBATCH -t 2-00:00:00
#SBATCH --mem 20240
#SBATCH -o output/{method}_en{noise_rate}_{noise_mode}_{model_type}.out
#SBATCH -e output/{method}_en{noise_rate}_{noise_mode}_{model_type}.err


source ~/.bashrc
conda activate base

i=0
noise_mode={noise_mode}
noise_rate={noise_rate}
method={method}
model_type={model_type}

python tm_train_enum.py \\
--dataset wiki \\
--noise_mode $noise_mode \\
--noise_rate $noise_rate \\
--model_type $model_type \\
--seed $i \\
--exp_name ../Noisywikihow_output/$noise_mode/enum/$model_type/wiki_$method/nr$noise_rate/ \\
--params_path wiki_em_params$method$noise_mode$noise_rate$model_type.json \\
--out_tmp wiki_em_out_tmp$method$noise_mode$noise_rate$model_type.json \\
--sub_script sbatch_wiki_em_sub$method$noise_mode$noise_rate$model_type.sh
'''

enum_sub_shell='''#!/bin/bash

i=0
noise_mode={noise_mode}
noise_rate={noise_rate}
method={method}
model_type={model_type}

python tm_train_hy_sub.py \\
--dataset wiki \\
--noise_mode $noise_mode \\
--noise_rate $noise_rate \\
--model_type $model_type \\
--seed $i \\
--exp_name ../Noisywikihow_output/$noise_mode/enum/$model_type/wiki_$method/nr$noise_rate/ \\
--params_path wiki_em_params$method$noise_mode$noise_rate$model_type.json \\
--out_tmp wiki_em_out_tmp$method$noise_mode$noise_rate$model_type.json \\
--sub_script sbatch_wiki_em_sub$method$noise_mode$noise_rate$model_type.sh
'''


# method = 'SEAL'
# [SLN]
# LOSS,
# [base, REWEIGHT, CT, SEAL, MIXUP, SR]
# noise_mode = 'sym'
# permute, cos, sib, iid
# mix, sym, idn
# method='STGN_GCE'
mode = 'nrun' #hy, nrun, enum, results
noise_rates = [0.6]
noise_modes = ['sym'] #[mix, sym, idn]
methods = ['base-lstm','MIXUP-lstm','REWEIGHT-lstm','SR-lstm','CT-lstm','SEAL-lstm']

def main():
    model_type = 'bart'
    if mode=='hy':
        for method in methods:
            for noise_rate in noise_rates:
                for noise_mode in noise_modes:
                # for noise_mode in ['mix']:
                    with open(f"sbatch_wiki_hy_params{method}{noise_mode}{noise_rate}{model_type}.sh","w") as f:
                        f.write(hy_shell.format(noise_rate=noise_rate, method=method, noise_mode=noise_mode,
                                                model_type=model_type))
                    with open(f"sbatch_wiki_hy_sub{method}{noise_mode}{noise_rate}{model_type}.sh","w") as f:
                        f.write(sub_shell.format(noise_rate=noise_rate, method=method, noise_mode=noise_mode,
                                                model_type=model_type))
    elif mode=='enum':
        for method in methods:
            for noise_rate in noise_rates:
                for noise_mode in noise_modes:
                # for noise_mode in ['mix']:
                    with open(f"sbatch_wiki_enum{method}{noise_mode}{noise_rate}{model_type}.sh", "w") as f:
                        f.write(enum_shell.format(noise_rate=noise_rate, method=method, noise_mode=noise_mode,
                                                model_type=model_type))
                    with open(f"sbatch_wiki_em_sub{method}{noise_mode}{noise_rate}{model_type}.sh", "w") as f:
                        f.write(enum_sub_shell.format(noise_rate=noise_rate, method=method, noise_mode=noise_mode,
                                                    model_type=model_type))
    elif mode=='results':
        for method in methods:
            for noise_mode in noise_modes:
            # for noise_mode in ['mix']:
                with open(f"sbatch_wiki_results_{method}{noise_mode}.sh", "w") as f:
                    f.write(results_shell.format(method=method, noise_mode=noise_mode,
                                            model_type=model_type))
    else:
        for method in methods:
            for noise_rate in noise_rates:
                for noise_mode in noise_modes:
                # for noise_mode in ['mix']:
                    with open(f"sbatch_wiki_hy_nrun{method}{noise_mode}{noise_rate}{model_type}.sh", "w") as f:
                        f.write(nrun_shell.format(noise_rate=noise_rate, method=method, noise_mode=noise_mode,
                                                model_type=model_type))

if __name__ == '__main__':
    main()