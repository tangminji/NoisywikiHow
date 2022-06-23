# 说明

## 文件
+ *enum.py 枚举参数调用sub文件，记录最优参数
+ *params.py 随机网格搜索参数（个人不推荐，太慢），调用sub文件，记录最优参数
+ *nrums.py 使用最优参数运行
+ *sub.py 由调参文件调用，执行一次并返回结果
+ shell_gen.py 快速生成脚本
+ cmd_args.py 参数文件

## 数据
+ data
    + wikihow Noisywikihow输入数据
        + noisy 噪声输入文件，事先生成. 记录了输入步骤和标签的编号。
            + 对sym,idn 格式(choosen_id,step_id,cat_id,noisy_label)即（编号，步骤id，类别id，噪声标签），输入时(x,y)用(step_id, noisy_label)
            + 对mix,tail,uncommon,neighbor 格式(choosen_id,step_id,noisy_id,cat_id)即（编号，步骤id，噪声步骤id，类别id），输入时(x,y)用(noisy_id, cat_id)
        + embedding 事先生成的各预训练模型的步骤embedding, 可以根据编号获得步骤的embedding
        + LT 长尾数据集
        + cat158.csv 158类事件意图标签信息，用于可视化以及t5模型训练
    + corrupt_index 噪声样本下标，可区分干净噪声样本，用于结果可视化
    
## 运行
参考运行配置：
+ bart, tesla_v100-pcie-32gb, batch_size=32, epochs=10, 运行约2.5h
+ lstm, tesla_v100-pcie-32gb, batch_size=32, epochs=10, 运行约0.25h

特殊的，Co-teaching要训练两个模型（约4h），有些模型在特定情况运行20epoch以充分训练保证达到峰值，SEAL默认运行4轮（40epoch），需适当延长运行时长。

## 运行脚本

```shell
#!/bin/bash
            
#SBATCH -J base_nrun0.4_mix_bart
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres gpu:tesla_v100-pcie-32gb:1
#SBATCH -t 1-00:00:00
#SBATCH --mem 20240
#SBATCH -e output/base_nrun0.4_mix_bart.err
#SBATCH -o output/base_nrun0.4_mix_bart.txt

source ~/.bashrc
conda activate base

noise_mode=mix
noise_rate=0.4
method=base
model_type=bart

for i in 0
do
python tm_train_hy_nruns.py \
  --dataset wiki \
  --noise_mode $noise_mode \
  --noise_rate $noise_rate \
  --model_type $model_type \
  --seed $i \
  --exp_name ../Noisywikihow_output/$noise_mode/nrun/$model_type/wiki_$method/nr$noise_rate/seed$i \
  --params_path best_params$method$noise_mode$noise_rate$model_type.json \
  --out_tmp wiki_out_tmp$method$noise_mode$noise_rate$model_type.json \
  --sub_script sbatch_wiki_hy_sub$method$noise_mode$noise_rate$model_type.sh
done

```

可适当修改，运行不同噪声下的各种模型。

+ noise_mode 噪声类型，可选['mix'(对应Noisywikihow), 'sym', 'idn', 'tail', 'uncommon', 'neighbor']
+ noise_rate 噪声率，一般可选[0.0, 0.1, 0.2, 0.4, 0.6], tail最多0.1, uncommon最多0.2
+ method 鲁棒学习方法，可选['base'(基线),'mixup', 'REWEIGHT'(Data Parameters), 'SR'(Sparse Regularization), 'CT'(Co-teaching), 'SEAL']
    + 可以在末尾添加'-lstm'使用2层bilstm来获得句子表示，用于后续分类。e.g. 'SR-lstm'
+ model_type 预训练模型的种类，可选['bert', 'roberta', 'xlnet', 'albert', 'bart', 'gpt2', 't5'], 其中**bart**表现最优

## 打包说明
代码(Noisywikihow的父目录下)：
> `zip -q -r Noisywikihow.zip Noisywikihow -x "Noisywikihow/data/wikihow/*" -x "Noisywikihow/data/corrupt_index/*"`

数据(Noisywikihow/data下)：
> `zip -q -r data.zip . -x "*.py"`

准备运行前，将data.zip拷贝到Noisywikihow/data下再解压