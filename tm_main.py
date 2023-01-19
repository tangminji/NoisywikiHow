import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from common.utils import log, AverageMeter, \
    compute_topk_accuracy, log_stats
from tqdm import tqdm
from tensorboard_logger import log_value
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data.wiki_dataset import get_wiki_train_and_val_loader, get_wiki_model_and_loss_criterion, WikiDataSetSoft, BartCLSWithEmbeds, get_wiki_tokenizer_and_label
import json
from hyperopt import STATUS_OK
import csv

from trainer import validate, train_others, train_soft, train_ct, \
    train_for_one_epoch, get_softmax_out, train_mixup, train_t5, validate_t5, train_cnlcu
from common.utils import get_class_inst_data_params_n_optimizer
from torch.utils.data import DataLoader

from common.losses import SRLoss

import pandas as pd

MD_CLASSES = {
    'wiki': (get_wiki_train_and_val_loader, get_wiki_model_and_loss_criterion)
}

def save_data(args, model, loader, criterion, mode='val'):
    """Save the outputs, targets, prob, loss for each sample"""
    # load the best model
    model_path = os.path.join(args.save_dir,'net.pt')
    log(args.logpath, "==> Save the prediction for each sample\n")
    log(args.logpath, f"Load model from {model_path}\n")
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()
    test_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')
    correct5 = AverageMeter('Acc@5', ':6.2f')
    t0 = time.time()
    with torch.no_grad():
        data_length = len(loader.dataset)
        outputs = torch.zeros(data_length, dtype=int)
        p_outputs = torch.zeros(data_length)
        targets = torch.zeros(data_length, dtype=int)
        p_targets = torch.zeros(data_length)
        losses = torch.zeros(data_length)

        for i, (data, target, index) in enumerate(loader):
            data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)
            output = model(**data)['logits']
            loss = criterion(output, target)
            prob = output.softmax(-1)
            p_outputs[index], outputs[index] = map(lambda x:x.cpu(), prob.max(-1))
            p_targets[index], targets[index] = map(lambda x:x.cpu(), [prob[range(len(output)),target], target])

            losses[index] = loss.cpu()
            loss = loss.mean()
            # measure accuracy and record loss
            test_loss.update(loss.item(), target.size(0))
            acc1, acc5 = compute_topk_accuracy(output, target, topk=(1, 5))
            correct1.update(acc1[0].item(), target.size(0))
            correct5.update(acc5[0].item(), target.size(0))
    log(args.logpath, 'Time for {}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.format(mode.capitalize(),
                                                                                      # 'Test'if mode=='test'else 'Val',
                                                                                      time.time() - t0,
                                                                                      correct1.avg, correct5.avg,
                                                                                      test_loss.avg))
    df = pd.DataFrame({
        'outputs': outputs,
        'p_outputs': p_outputs,
        'targets': targets,
        'p_targets': p_targets,
        'losses': losses
    }).round(3)
    save_path = os.path.join(args.log_dir,f'{mode}_results.csv')
    df.to_csv(save_path, index=None)
    log(args.logpath, f'Successfully saved the {mode} results at {save_path}!\n')


def main(args, params):
    """The train and test process"""
    # Keep track of evals

    loaders, mdl_loss = MD_CLASSES[args.dataset]
    # Create model
    net, criterion, criterion_val = mdl_loss(args)

    train_loader, val_loader, test_loader, noisy_ind, clean_ind = loaders(args)
    train_length = len(train_loader.dataset)

    # TODO noisy_index, clean_index, numpy format
    args.noisy_ind = np.array(noisy_ind)
    args.clean_ind = np.array(clean_ind)

    # update perturb variance, dynamic sigma for each sample
    parameters = list(filter(lambda x: x.requires_grad, net.parameters()))

    cudnn.benchmark = True
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)

    # Training
    global_t0 = time.time()
    global_iter = 0
    global test_best1
    global test_best5
    global val_best1, val_best5, test_at_best1, test_at_best5
    test_best1 = 0
    test_best5 = 0
    val_best1 = 0
    val_best5 = 0
    test_at_best1 = 0
    test_at_best5 = 0
    res_lst = []

    # Data parameter
    if 'REWEIGHT' in args.exp_name:
        (class_parameters, inst_parameters,
        optimizer_class_param, optimizer_inst_param) = get_class_inst_data_params_n_optimizer(
                                                            args=args,
                                                            nr_classes=args.num_class,
                                                            nr_instances=train_length,
                                                            device='cuda'
                                                            )
        config = {}
        config['clamp_inst_sigma'] = {}
        config['clamp_inst_sigma']['min'] = np.log(1/20)
        config['clamp_inst_sigma']['max'] = np.log(20)
        config['clamp_cls_sigma'] = {}
        config['clamp_cls_sigma']['min'] = np.log(1/20)
        config['clamp_cls_sigma']['max'] = np.log(20)

    # Co-teaching
    if 'CT' in args.exp_name or 'CNLCU' in args.exp_name:
        net2, criterion2, _ = mdl_loss(args)
        parameters = list(filter(lambda x: x.requires_grad, net2.parameters()))
        optimizer2 = torch.optim.AdamW(parameters, lr=args.lr)
        if args.forget_rate is None:
            forget_rate=args.noise_rate
        else:
            forget_rate=args.forget_rate
        rate_schedule = np.ones(args.epochs)*forget_rate # max: forget_rate
        rate_schedule[:args.ct_num_gradual] = np.linspace(0, forget_rate**args.exponent, args.ct_num_gradual) # linear
    
    # SEAL
    if 'SEAL' in args.exp_name:
        # loader for get predictions on train set
        softmax_loader = DataLoader(train_loader.dataset,batch_size=args.test_batch_size, shuffle=False)
        softmax_out_avg = torch.zeros([train_length, args.num_class]) #torch:float32

    # MIXUP
    if 'MIXUP' in args.exp_name:
        net = BartCLSWithEmbeds(net).to(args.device)

    # SR
    if 'SR' in args.exp_name:
        criterion = SRLoss(criterion=criterion, lamb=args.lamb, tau=args.tau, p=args.normp, reduction='none')

    # TODO Other Methods
    if 'CNLCU' in args.exp_name:
        # ? 为0应该就无法区分了
        args.co_lambda_plan = np.zeros(args.epochs)
        gradient_range = min(args.epoch_decay_start, args.epochs)
        args.co_lambda_plan[:gradient_range] = args.co_lambda * np.linspace(1, 0, gradient_range)
        noise_or_not = np.ones(train_length)
        noise_or_not[noisy_ind] = 0 
        # The CNLCU implement: (1 for clean, 0 for )
        #   noise_or_not = np.transpose(self.train_labels)==np.transpose(self.train_labels_o) 

    # show result only
    if args.show_result:
        show_train_loader = DataLoader(train_loader.dataset,batch_size=args.test_batch_size, shuffle=False)
        save_data(args, net, show_train_loader,criterion_val, mode='train')
        save_data(args, net, val_loader, criterion_val, mode='val')
        save_data(args, net, test_loader, criterion_val, mode='test')
        return

    # TODO For t5
    if args.model_type == 't5':
        tokenizer, cat_token, cat_labels = get_wiki_tokenizer_and_label(args)

    # epoch 0: test only
    # epoch 1~epochs: train + test
    for epoch in range(0, args.epochs + 1):
        args.cur_epoch = epoch
        # Training
        if epoch > 0:
            # TODO Be careful: Use if-elif to ensure only train once
            # t5 training
            if args.model_type == 't5':
                global_iter, train_loss, train_acc1, train_acc5 = \
                    train_t5(args, net, train_loader, optimizer, criterion, global_iter, epoch, tokenizer, cat_token, cat_labels)
            # SEAL and finished the first round, use soft-label to train
            elif 'SEAL' in args.exp_name and epoch > args.epoch_round:
                if epoch % args.epoch_round==1:
                    softmax_out_avg /= args.epoch_round
                    torch.save(softmax_out_avg, os.path.join(args.save_dir,'soft_label.pt')) # save soft-label temporarily
                    train_soft_dataset = WikiDataSetSoft(train_loader.dataset, softmax_out_avg)
                    print(f"Generate new softloader at epoch {epoch}")
                    train_soft_loader = DataLoader(train_soft_dataset, batch_size=args.batch_size,shuffle=True)
                    softmax_out_avg = torch.zeros([train_length, args.num_class])
                global_iter, train_loss, train_acc1, train_acc5 = \
                    train_soft(args, net, train_soft_loader, optimizer, criterion, global_iter, epoch)

            elif 'CT' in args.exp_name:
                global_iter, train_loss, train_acc1, train_acc5 = \
                    train_ct(args, net, net2, optimizer, optimizer2, criterion, criterion2, train_loader, global_iter, epoch, p_keep=1-rate_schedule[epoch-1]) 
            # ICLR22
            elif 'CNLCU' in args.exp_name:
                if epoch % args.time_step == 1:
                    print('Time step initializing...')
                    before_loss_1 = 0.0 * np.ones((train_length, 1))
                    before_loss_2 = 0.0 * np.ones((train_length, 1))
                    sn_1 = torch.from_numpy(np.ones((train_length, 1)))
                    sn_2 = torch.from_numpy(np.ones((train_length, 1)))
                    # args, model1, model2, optimizer1, optimizer2, criterion1, criterion2, loader, global_iter, epoch, before_loss_1, before_loss_2, sn_1, sn_2, noise_or_not, forget_rate
                global_iter, train_loss, train_acc1, train_acc5, before_loss_1_list, before_loss_2_list, ind_1_update_list, ind_2_update_list= train_cnlcu(args, net, net2, optimizer, optimizer2, train_loader, global_iter, epoch, before_loss_1, before_loss_2, sn_1, sn_2, noise_or_not, rate_schedule[epoch-1])
                # TODO FIXED
                before_loss_1_, before_loss_2_ = torch.tensor(before_loss_1_list).numpy().astype(float), torch.tensor(before_loss_2_list).numpy().astype(float)

                before_loss_1_numpy = np.zeros((train_length, 1))
                before_loss_2_numpy = np.zeros((train_length, 1))
                num = before_loss_1_.shape[0]
                before_loss_1_numpy[:num], before_loss_2_numpy[:num] = before_loss_1_[:, np.newaxis], before_loss_2_[:, np.newaxis]
                before_loss_1 = np.concatenate((before_loss_1, before_loss_1_numpy), axis=1)
                before_loss_2 = np.concatenate((before_loss_2, before_loss_2_numpy), axis=1)
                # save the selection history
                all_zero_array_1, all_zero_array_2 = np.zeros((train_length, 1)), np.zeros((train_length, 1))
                all_zero_array_1[np.array(ind_1_update_list)] = 1
                all_zero_array_2[np.array(ind_2_update_list)] = 1
            
                sn_1 += torch.from_numpy(all_zero_array_1)
                sn_2 += torch.from_numpy(all_zero_array_2)
                    
            elif 'MIXUP' in args.exp_name:
                global_iter, train_loss, train_acc1, train_acc5 = \
                    train_mixup(args, net, train_loader, optimizer, criterion, global_iter, epoch)
            elif 'REWEIGHT' in args.exp_name:
                global_iter, train_loss, train_acc1, train_acc5 = \
                    train_for_one_epoch(
                            args=args,
                            loader=train_loader,
                            model=net,
                            criterion=criterion,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_iter=global_iter,
                            optimizer_data_parameters=(optimizer_class_param, optimizer_inst_param),
                            data_parameters=(class_parameters, inst_parameters),
                            config=config)
            else:
                global_iter, train_loss, train_acc1, train_acc5 = \
                    train_others(args, net, train_loader, optimizer, criterion, global_iter, epoch)

            # TODO Train function for Other Methods


            # TODO After training
            if 'SEAL' in args.exp_name:
                softmax_out_avg += get_softmax_out(args, net, softmax_loader)
            if 'SR' in args.exp_name:
                if args.freq>0 and epoch % args.freq == 0: # every 'freq' epochs, update 'lambda' once by 'rho' 
                    criterion.update_lamb(args.rho)

        # Testing
        # t5 validate
        if args.model_type == 't5':
            val_loss, val_acc1, val_acc5 = validate_t5(args, net, val_loader, criterion_val, epoch, tokenizer, cat_token, cat_labels, mode='val')
            test_loss, test_acc1, test_acc5 = validate_t5(args, net, test_loader, criterion_val, epoch, tokenizer, cat_token, cat_labels, mode='test')
        else:
            val_loss, val_acc1, val_acc5 = validate(args, net, val_loader, criterion_val, epoch,
                                                    mode='val')
            test_loss, test_acc1, test_acc5 = validate(args, net, test_loader, criterion_val, epoch,
                                                    mode='test')

        # Save checkpoint.
        if val_acc1 > val_best1:
            val_best1 = val_acc1
            test_at_best1 = test_acc1
            test_at_best5 = test_acc5
            torch.save(net.state_dict(), os.path.join(args.save_dir, 'net.pt'))
        
        if val_acc5 > val_best5:
            val_best5 = val_acc5
        
        if test_acc1 > test_best1:
            test_best1 = test_acc1
            
        if test_acc5 > test_best5:
            test_best5 = test_acc5

        # epoch 0: Test only, havn't train yet
        if epoch == 0:
            continue
        res_lst.append((train_acc1, train_acc5, val_acc1, val_acc5, test_acc1, test_acc5, train_loss, val_loss, test_loss))

        # Logging
        # data parameter
        if 'REWEIGHT' in args.exp_name:
            if len(noisy_ind) > 0:
                log_stats(data=torch.exp(inst_parameters[noisy_ind]),
                        name='epoch_stats_corrupt_inst_parameter',
                        step=epoch)
            if len(clean_ind) > 0:
                log_stats(data=torch.exp(inst_parameters[clean_ind]),
                        name='epoch_stats_clean_inst_parameter',
                        step=epoch)
        
        # TODO Other logging steps

    run_time = time.time() - global_t0
    # save 3 types of acc
    # record best_acc/best_mae
    with open(os.path.join(args.log_dir, 'acc_loss_results.txt'), 'w', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerows(res_lst)

    # The performance on last five epochs
    stable_acc1 = sum([x[4] for x in res_lst[-5:]]) / 5
    stable_acc5 = sum([x[5] for x in res_lst[-5:]]) / 5

    # Val_best Test_at_val_best Stable_test_acc
    with open(os.path.join(args.log_dir, 'best_results.txt'), 'w') as outfile:
        outfile.write(f'{test_at_best1}\t{test_at_best5}\t{test_best1}\t{test_best5}\t{stable_acc1}\t{stable_acc5}')
    log(args.logpath, '\nTest Acc:\nAt Best Val:\t Acc1: {}\t Acc5: {}\nAt Best Test:\t Acc1: {}\t  Acc5: {}\n Stable:\t Acc1: {}\t Acc5: {}'.format(test_at_best1, test_at_best5, test_best1, test_best5, stable_acc1, stable_acc5))
    log(args.logpath, '\nTotal Time: {:.1f}s.\n'.format(run_time))

    loss = - test_best1
    return {'loss': loss, 'test_at_best_top1': test_at_best1, 'test_at_best_top5': test_at_best5, 'test_top1': test_best1, 'test_top5':  test_best5,
            'stable_acc_top1': stable_acc1, 'stable_acc_top5': stable_acc5,
            'params': params, 'train_time': run_time, 'status': STATUS_OK}