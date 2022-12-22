# NoisywikiHow

## Code
+ tm_train_hy_nrums.py  The entry code. Set the Optimal parameters for each methods.
+ tm_main.py            The main training structure. Called by `tm_train_hy_nruns.py`.
+ trainer.py            Different training methods. Called by `tm_main.py`.
+ cmd_args.py           The command arguments.
+ data/wiki_dataset.py  Dataloader and Models for experiments.

## Data
+ data
    + wikihow           NoisywikiHow Dataset
        + noisy/        The input folder
            + train.csv                         The clean train data. Format `choosen_id, step_id, cat_id, step, cat`
            + val_test.csv                      The clean validation and test data, will split into `val.csv`, `test.csv`
            + val.csv                           The clean validation data. Format `choosen_id, step_id, cat_id, step, cat`
            + test.csv                          The clean test data. Format `choosen_id, step_id, cat_id, step, cat`
            + mix_{0.1,0.2,0.4,0.6}.csv         Noisywikihow train data with noise. Format `choosen_id, step_id, noisy_id, cat_id, step, cat, noisy_step, noisy_cat, noisy_label`.
                +   Take `(noisy_id, cat_id)` as input.
            + sym_{0.1,0.2,0.4,0.6}.csv         Train data with symmetric noise. Format `choosen_id,step_id,cat_id,noisy_label,step,cat,noisy_cat`.
                +   Take `(step_id, noisy_label)` as input.
            + {tail,uncommon,neighbor}_0.1.csv  Train data with different noise sources. Format is the same as `mix_0.1.csv`.
        + cat158.csv    The choosen 158 event intention classes.
        + split_val_test.py     The code for splitting val set and test set.

### Fields description

| Field         | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `choosen_id`  | **global id** for this sample. range from 0~89142            |
| `step`        | The real step text.                                          |
| `step_id`     | The id for `step`                                            |
| `cat`         | The real event intention category text.                      |
| `cat_id`      | The id for `cat`                                             |
| `noisy_step`  | The **corrupted** step text with **certain noise rate**. It may be agree with `step`,  may not either. |
| `noisy_cat`   | For noise on labels(`sym`, etc.):  The **corrupted** event intention label with **certain noise rate**. It may be agree with `cat`,  may not either.<br />For similar noisy steps(`mix`,`tail`,`uncommon`,`neighbor`):  The event intention category correspond to `noisy step`. |
| `noisy_label` | The id for `noisy_cat`. `-1` for the category out of given 158 classes (OOV class). |
    
## Running
+ bart, tesla_v100-pcie-32gb, batch_size=32: 2.5h/10epochs
    + Co-teaching: 4h/10epochs, SEAL run 40epochs
+ lstm, tesla_v100-pcie-32gb, batch_size=32: 0.25h/10epochs
    + Co-teaching: 0.4h/10epochs, SEAL run 40epochs


## Shell

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

You can change arguments for different experiments.

+ noise_mode 
    + You can choose `['mix'(NoisyWikihow), 'sym', 'tail', 'uncommon', 'neighbor']`
+ noise_rate
    + For `'mix','sym'`, you can choose `[0.0, 0.1, 0.2, 0.4, 0.6]`.
    + For `'tail', 'uncommon', 'neighbor'`, you must choose `[0.1]`.
+ method
    + You can choose `['base'(baseline),'mixup', 'REWEIGHT'(Data Parameters), 'SR'(Sparse Regularization), 'CT'(Co-teaching), 'SEAL']`
    + You can add `-lstm` at the end to use 2-layer BiLSTM instead of pretrained model to get the sentence representation。e.g. 'SR-lstm'
+ model_type
    + You can choose `['bert', 'roberta', 'xlnet', 'albert', 'bart', 'gpt2', 't5']`. Model `bart` has the best performance.