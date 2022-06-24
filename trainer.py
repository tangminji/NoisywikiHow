"""
Different training methods
"""
import torch,time
from common.utils import log, AverageMeter, \
    compute_topk_accuracy, checkpoint, log_intermediate_iteration_stats, log_stats, log_intermediate_iteration_stats_parameter
from common.utils import get_data_param_for_minibatch, apply_weight_decay_data_parameters, clamp_data_parameters
from torch.nn.utils import clip_grad_norm_
from tensorboard_logger import log_value
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import accuracy_score
import pandas as pd

import os
import csv

def save_predict(save_dir, predict, epoch):
    """
    save prediction for each sample in one epoch
    """
    save_path = save_dir + '/predict_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        loss_ep = {'epoch:{}'.format(epoch): predict.tolist()}
        outfile.write('{}{}'.format(loss_ep, '\n'))


def save_loss(save_dir, loss, epoch):
    """
    save loss for each sample in one epoch
    """
    save_path = save_dir + '/loss_each_sample_one_eps.txt'
    with open(save_path, 'a+') as outfile:
        loss_ep = {'epoch:{}'.format(epoch): loss.tolist()}
        outfile.write('{}{}'.format(loss_ep, '\n'))

# TODO log detail stats
# There may be some bugs. 
# Unused at present
def log_epoch_predict_and_loss(args, loss_parameters, predictions, p_target, p_max, epoch):
    save_loss(args.save_dir, loss_parameters, epoch)
    save_predict(args.save_dir, predictions, epoch)

    loss_clean, loss_corrupt = loss_parameters[args.clean_ind], loss_parameters[args.noisy_ind]
    p_target_clean, p_target_corrupt = p_target[args.pred_ind], p_target[args.noisy_ind]
    p_max_clean, p_max_corrupt = p_max[args.pred_ind], p_max[args.noisy_ind]
    log_stats(data=loss_clean, name='loss_clean', step=epoch)
    log_stats(data=p_target_clean, name='p_target_clean', step=epoch)
    log_stats(data=p_max_clean, name='p_max_clean', step=epoch)
    if len(args.noisy_ind)>0:
        log_stats(data=loss_corrupt, name='loss_corrupt',step=epoch)
        log_stats(data=p_target_corrupt, name='p_target_corrupt', step=epoch)
        log_stats(data=p_max_corrupt, name='p_max_corrupt', step=epoch)

    logfile = os.path.join(args.save_dir, 'p_target.csv')
    with open(logfile, 'a+', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerows(p_target.tolist())
    logfile = os.path.join(args.save_dir, 'p_max.csv')
    with open(logfile, 'a+', newline='') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerows(p_max.tolist())
    


# data_parameter
def train_for_one_epoch(args,
                        loader,
                        model,
                        criterion,
                        optimizer,
                        epoch,
                        global_iter,
                        optimizer_data_parameters,
                        data_parameters,
                        config):
    """Train model for one epoch on the train set.

    Args:
        args (argparse.Namespace):
        loader (torch.utils.data.dataloader): dataloader for train set.
        model (torch.nn.Module): DNN model.
        criterion (torch.nn.modules.loss): cross entropy loss.
        optimizer (torch.optim.SGD): optimizer for model parameters.
        epoch (int): current epoch.
        global_iter (int): current iteration count.
        optimizer_data_parameters (tuple SparseSGD): SparseSGD optimizer for class and instance data parameters.
        data_parameters (tuple of torch.Tensor): class and instance level data parameters.
        config (dict): config file for the experiment.

    Returns:
        global iter (int): updated iteration count after 1 epoch.
    """

    # Initialize counters
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')
    correct5 = AverageMeter('Acc@5', ':6.2f')
    t0 = time.time()
    loss_parameters = torch.zeros(len(loader.dataset))
    predictions = torch.zeros(len(loader.dataset), args.num_class)
    # Unpack data parameters
    optimizer_class_param, optimizer_inst_param = optimizer_data_parameters
    class_parameters, inst_parameters = data_parameters

    for i, (data, target, index) in enumerate(loader):
        global_iter = global_iter + 1
        data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)

        # Flush the gradient buffer for model and data-parameters
        optimizer.zero_grad()
        if args.learn_class_parameters:
            optimizer_class_param.zero_grad()
        if args.learn_inst_parameters:
            optimizer_inst_param.zero_grad()

        # Compute logits
        output = model(**data)['logits']
        
        if args.learn_class_parameters or args.learn_inst_parameters:
            # Compute data parameters for instances in the minibatch
            class_parameter_minibatch = class_parameters[target]
            inst_parameter_minibatch = inst_parameters[index]
            data_parameter_minibatch = get_data_param_for_minibatch(
                                            args,
                                            class_param_minibatch=class_parameter_minibatch,
                                            inst_param_minibatch=inst_parameter_minibatch)

            # Compute logits scaled by data parameters
            output = output / data_parameter_minibatch

        # TODO You can record p_target, p_max.
        # TODO We may compute predictions after the logits changed
        # Add
        predictions[index] = F.softmax(output).detach().cpu()


        loss = criterion(output, target)
        loss_parameters[index] = loss.detach().cpu()
        loss = loss.mean()

        
        # Apply weight decay on data parameters
        if args.learn_class_parameters or args.learn_inst_parameters:
            loss = apply_weight_decay_data_parameters(args, loss,
                                                            class_parameter_minibatch=class_parameter_minibatch,
                                                            inst_parameter_minibatch=inst_parameter_minibatch)

        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if args.learn_class_parameters:
            optimizer_class_param.step()
        if args.learn_inst_parameters:
            optimizer_inst_param.step()

        # Clamp class and instance level parameters within certain bounds
        if args.learn_class_parameters or args.learn_inst_parameters:
            clamp_data_parameters(args, class_parameters, config, inst_parameters)

        # Measure accuracy and record loss
        train_loss.update(loss.item(), target.size(0))
        acc1, acc5 = compute_topk_accuracy(output, target, topk=(1, 5))
        correct1.update(acc1[0].item(), target.size(0))
        correct5.update(acc5[0].item(), target.size(0))
        # # Log stats for data parameters and loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats_parameter(args, class_parameters, epoch,
                                                   global_iter, inst_parameters,
                                                   train_loss, top1=correct1, top5=correct5)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(args.logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.
        format(epoch, args.epochs, time.time() - t0, correct1.avg, correct5.avg, train_loss.avg))

    log_value('train/accuracy_1', correct1.avg, step=epoch)
    log_value('train/accuracy_5', correct5.avg, step=epoch)

    save_loss(args.save_dir, loss_parameters, epoch)
    save_predict(args.save_dir, predictions, epoch)

    return global_iter, train_loss.avg, correct1.avg, correct5.avg


def onehotCE(pred, y_hot):
    """Compute CrossEntropy loss, (B,C) x (B,C) -> (B)"""
    return -torch.sum(F.log_softmax(pred, dim=1) * y_hot, dim=1)

# compute sotmax label
def get_softmax_out(args, model, loader):
    softmax_out = []
    model.eval()
    with torch.no_grad():
        for data, *_ in loader:
            data = {k: v.to(args.device) for k, v in data.items()}
            output = model(**data)['logits']
            sm = F.softmax(output, dim=-1).cpu() # Must save on cpu, avoid CUDAError
            softmax_out.append(sm)
    return torch.cat(softmax_out) # float32

def train_soft(args, model, loader, optimizer, criterion, global_iter, epoch):
    """
    Train model for SEAL with averaged soft labels.
    
    Note: SEAL use the given labels instead in the first round.
    """
    print(f"Soft Training Epoch{epoch}:")
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')  # for classification
    correct5 = AverageMeter('Acc@5', ':6.2f')  # for classification
    t0 = time.time()

    loss_parameters = torch.zeros(len(loader.dataset))
    predictions = torch.zeros(len(loader.dataset), args.num_class)
    # target_soft: like one_hot vector, [B,C]
    for i, (data, target, target_soft, index) in enumerate(loader):
        global_iter += 1
        # similar to global variable
        args.index = index
        data, target, target_soft = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device), target_soft.to(args.device)

        output = model(**data)['logits']

        predictions[index] = F.softmax(output).detach().cpu()

        loss = onehotCE(output, target_soft)

        loss_parameters[index] = loss.detach().cpu()

        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        # Measure accuracy and record loss
        train_loss.update(loss.item(), target.size(0))
        acc1, acc5 = compute_topk_accuracy(output, target, topk=(1, 5))
        correct1.update(acc1[0].item(), target.size(0))
        correct5.update(acc5[0].item(), target.size(0))
        # Log loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct1, top5=correct5)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(args.logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.
        format(epoch, args.epochs, time.time() - t0, correct1.avg, correct5.avg, train_loss.avg))

    log_value('train/accuracy_1', correct1.avg, step=epoch)
    log_value('train/accuracy_5', correct5.avg, step=epoch)

    save_loss(args.save_dir, loss_parameters, epoch)
    save_predict(args.save_dir, predictions, epoch)

    return global_iter, train_loss.avg, correct1.avg, correct5.avg


def train_ct(args, model1, model2, optimizer1, optimizer2, criterion1, criterion2, loader, global_iter, epoch, p_keep):
    """
    Train 2 models for Co-teaching simultaneously, take the output of the first model as the results.

    Note: We don't record loss or pred for each sample in order to save space
    """
    model1.train(), model2.train()
    train_loss1 = AverageMeter('Loss', ':.4e')
    train_loss2 = AverageMeter('Loss', ':.4e')
    correct1_1 = AverageMeter('Acc@1', ':6.2f')
    correct5_1 = AverageMeter('Acc@5', ':6.2f')
    correct1_2 = AverageMeter('Acc@1', ':6.2f')
    correct5_2 = AverageMeter('Acc@5', ':6.2f')
    
    # The small loss sample selected by models
    small_loss1 = torch.zeros(len(loader.dataset))
    small_loss2 = torch.zeros(len(loader.dataset))

    t0 = time.time()
    for i, (data, target, index) in enumerate(loader):
        n_keep = round(p_keep*target.size(0))
        global_iter += 1
        args.index = index
        data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)
        output1, output2 = model1(**data)['logits'], model2(**data)['logits']
        loss1, loss2 = criterion1(output1, target), criterion2(output2, target)
        # selecting #n_keep small loss instances
        _, index1 = torch.sort(loss1.detach())
        _, index2 = torch.sort(loss2.detach())
        index1, index2 = index1[:n_keep], index2[:n_keep] # the small loss sample

        # record them
        with torch.no_grad():
            small_loss1[index[index1]]=1.0
            small_loss2[index[index2]]=1.0

        # taking a optimization step
        optimizer1.zero_grad()
        loss1[index2].mean().backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2[index1].mean().backward()
        optimizer2.step()

        # loss1 = loss1.sum()
        # loss2 = loss2.sum()
        # TODO use mean, not sum
        loss1 = loss1.mean()
        loss2 = loss2.mean()

        train_loss1.update(loss1.item(), target.size(0))
        acc1, acc5 = compute_topk_accuracy(output1, target, topk=(1, 5))
        correct1_1.update(acc1[0].item(), target.size(0))
        correct5_1.update(acc5[0].item(), target.size(0))

        train_loss2.update(loss2.item(), target.size(0))
        acc1, acc5 = compute_topk_accuracy(output2, target, topk=(1, 5))
        correct1_2.update(acc1[0].item(), target.size(0))
        correct5_2.update(acc5[0].item(), target.size(0))

        # Log loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss1, top1=correct1_1, top5=correct5_1,
                title="train_iteration_stats/net1")
            log_intermediate_iteration_stats(epoch, global_iter, train_loss2, top1=correct1_2, top5=correct5_2,
                title="train_iteration_stats/net2")

    # Print and log stats for the epoch
    log_value('train/net1/loss', train_loss1.avg, step=epoch)
    log(args.logpath, 'Net1: Time for Train-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.
        format(epoch, args.epochs, time.time() - t0, correct1_1.avg, correct5_1.avg, train_loss1.avg))

    log_value('train/net1/accuracy_1', correct1_1.avg, step=epoch)
    log_value('train/net1/accuracy_5', correct5_1.avg, step=epoch)

    log_value('train/net2/loss', train_loss2.avg, step=epoch)
    log(args.logpath, 'Net2: Time for Train-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.
        format(epoch, args.epochs, time.time() - t0, correct1_2.avg, correct5_2.avg, train_loss2.avg))

    log_value('train/net1/accuracy_1', correct1_2.avg, step=epoch)
    log_value('train/net1/accuracy_5', correct5_2.avg, step=epoch)

    # clean rate in small_loss sample
    clean_rate1 = small_loss1[args.clean_ind].sum() / small_loss1.sum()
    clean_rate2 = small_loss2[args.clean_ind].sum() / small_loss2.sum()
    
    log_value('small_loss/clean1', clean_rate1, step=epoch)
    log_value('small_loss/clean2', clean_rate2, step=epoch)

    return global_iter, train_loss1.avg, correct1_1.avg, correct5_1.avg



# Mixup
def mixup_data(inputs_embeds, decoder_inputs_embeds, y, alpha=1.0):
    """
    Generate virtual feature-target vectors.
    
    x = lam*x_a + (1-lam)*x_b
    
    y = lam*y_a + (1-lam)*y_b
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = inputs_embeds.size()[0]
    index = torch.randperm(batch_size)
    inputs_embeds = lam * inputs_embeds + (1-lam)*inputs_embeds[index]
    decoder_inputs_embeds = lam * decoder_inputs_embeds + (1-lam)*decoder_inputs_embeds[index]
    y_a, y_b = y, y[index]
    return inputs_embeds, decoder_inputs_embeds, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for vicinity distribution"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(args, model, loader, optimizer, criterion, global_iter, epoch):
    """
    Train model for mixup on vicinity distribution
    """
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')  # for classification
    correct5 = AverageMeter('Acc@5', ':6.2f')  # for classification
    t0 = time.time()

    loss_parameters = torch.zeros(len(loader.dataset))
    predictions = torch.zeros(len(loader.dataset), args.num_class)
    for i, (data, target, index) in enumerate(loader):
        global_iter += 1
        # similar to global variable
        args.index = index
        data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)

        # use the first sample to find [EOS]
        input_ids = data['input_ids']
        inputs_embeds, decoder_inputs_embeds = model.generate_embeds(input_ids)
        attention_mask = data['attention_mask']
        # mixup
        inputs_embeds, decoder_inputs_embeds, y_a, y_b, lam = mixup_data(inputs_embeds, decoder_inputs_embeds, target, args.alpha)

        output = model(input_ids=input_ids, inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,attention_mask=attention_mask)['logits']

        # ADD
        predictions[index] = F.softmax(output).detach().cpu()

        loss = mixup_criterion(criterion, output, y_a, y_b, lam=lam)

        # ADD
        loss_parameters[index] = loss.detach().cpu()

        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        # Measure accuracy and record loss
        train_loss.update(loss.item(), target.size(0))

        acc1, acc5 = compute_topk_accuracy(output, y_a, topk=(1, 5))
        acc1b, acc5b = compute_topk_accuracy(output, y_b, topk=(1, 5))
        
        correct1.update(lam * acc1[0].item() + (1-lam)*acc1b[0].item(), target.size(0))
        correct5.update(lam * acc5[0].item() + (1-lam)*acc5b[0].item(), target.size(0))

        # Log loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct1, top5=correct5)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(args.logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.
        format(epoch, args.epochs, time.time() - t0, correct1.avg, correct5.avg, train_loss.avg))

    log_value('train/accuracy_1', correct1.avg, step=epoch)
    log_value('train/accuracy_5', correct5.avg, step=epoch)

    save_loss(args.save_dir, loss_parameters, epoch)
    save_predict(args.save_dir, predictions, epoch)

    return global_iter, train_loss.avg, correct1.avg, correct5.avg


def train_others(args, model, loader, optimizer, criterion, global_iter, epoch):
    """
    train model
    """
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')  # for classification
    correct5 = AverageMeter('Acc@5', ':6.2f')  # for classification
    t0 = time.time()

    loss_parameters = torch.zeros(len(loader.dataset))
    predictions = torch.zeros(len(loader.dataset), args.num_class)
    # loss_lst = TDigest()
    loss_lst = []
    for i, (data, target, index) in enumerate(loader):
        global_iter += 1
        # similar to global variable
        args.index = index
        # if len(target.size()) == 1:
        #     target = torch.zeros(target.size(0), args.num_class).scatter_(1, target.view(-1, 1),
        #                                                                   1)  # convert label to one-hot
        data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)

        output = model(**data)['logits']

        # ADD
        predictions[index] = F.softmax(output).detach().cpu()

        loss = criterion(output, target)
        loss_parameters[index] = loss.detach().cpu()

        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        # Measure accuracy and record loss
        train_loss.update(loss.item(), target.size(0))
        acc1, acc5 = compute_topk_accuracy(output, target, topk=(1, 5))
        correct1.update(acc1[0].item(), target.size(0))
        correct5.update(acc5[0].item(), target.size(0))

        # Log loss every few iterations
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct1, top5=correct5)

    # Print and log stats for the epoch
    log_value('train/loss', train_loss.avg, step=epoch)
    log(args.logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.
        format(epoch, args.epochs, time.time() - t0, correct1.avg, correct5.avg, train_loss.avg))

    log_value('train/accuracy_1', correct1.avg, step=epoch)
    log_value('train/accuracy_5', correct5.avg, step=epoch)

    save_loss(args.save_dir, loss_parameters, epoch)
    save_predict(args.save_dir, predictions, epoch)

    return global_iter, train_loss.avg, correct1.avg, correct5.avg


def train_t5(args, model, loader, optimizer, criterion, global_iter, epoch, tokenizer, cat_token, cat_labels):
    """
    Train t5 model. For t5, we force t5 to generate the event intention label, which it did in its origin paper.

    Note: For fast training, we don't search the generated label text, which consume too much time
    so we omit acc1,acc5 on train set
    """
    model.train()
    train_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')
    correct5 = AverageMeter('Acc@5', ':6.2f')
    t0 = time.time()
    
    for i, (data, target, *_) in enumerate(loader):
        global_iter += 1
        data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)
        labels = cat_token[target] # label token，pad with -100
        loss = model(**data, labels=labels)['loss']

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        
        train_loss.update(loss.item(), target.size(0))
        # omit acc1,acc5
        acc1 = 0 
        acc5 = acc1 
        correct1.update(acc1, target.size(0))
        correct5.update(acc5, target.size(0))
        if i % args.print_freq == 0:
            log_intermediate_iteration_stats(epoch, global_iter, train_loss, top1=correct1, top5=correct5)
    
    log_value('train/loss', train_loss.avg, step=epoch)
    log(args.logpath, 'Time for Train-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.
        format(epoch, args.epochs, time.time() - t0, correct1.avg, correct5.avg, train_loss.avg))
    return global_iter, train_loss.avg, correct1.avg, correct5.avg


def compute_top5(true, preds):
    acc1, acc5 = 0.0, 0.0
    n = len(true)
    for t, p in zip(true, preds):
        if t in p[:5]:
            acc5 += 1
            if t == p[0]:
                acc1 += 1
    acc1 = acc1/n * 100.0
    acc5 = acc5/n * 100.0
    return acc1, acc5

def validate_t5(args, model, loader, criterion, epoch, tokenizer, cat_token, cat_labels, mode='val'):
    """
    Evaluates t5 model on validation/test set and logs score on tensorboard.

    For t5, use Beam Search to generate 5 best the event intention labels
    """
    test_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')
    correct5 = AverageMeter('Acc@5', ':6.2f')
    # switch to evaluate mode
    model.eval()
    t0 = time.time()
    y_label, y_predict = [],[]
    with torch.no_grad():
        for i, (data, target, *_) in enumerate(loader):
            data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)
            labels = cat_token[target] # label token，pad with -100
            loss = model(**data, labels=labels)['loss']
            # Beam Search, num_beams=5, Top 5 answers
            generated_ids = model.generate(
                    **data,
                    num_beams=5,
                    num_return_sequences=5,
                    max_length=15
                    )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            true = [cat_labels[c] for c in target]
            text = [preds[p:p+5] for p in range(0, len(preds), 5)] # top5
            preds = [t[0] for t in text] # top1

            y_predict.extend(preds)
            y_label.extend(true)
            # measure accuracy and record loss
            test_loss.update(loss.item(), target.size(0))
            acc1, acc5 = compute_top5(true, text)
            correct1.update(acc1, target.size(0))
            correct5.update(acc5, target.size(0))
    log(args.logpath, 'Time for {}-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.format(mode.capitalize(),
                                                                                      epoch, args.epochs,
                                                                                      time.time() - t0,
                                                                                      correct1.avg, correct5.avg,
                                                                                      test_loss.avg))
    # Save the predictions
    df = pd.DataFrame({'label':y_label,'predict':y_predict})
    df.to_csv(f'{args.save_dir}/{mode}_epoch{epoch}.csv')
    
    log_value('{}/loss'.format(mode), test_loss.avg, step=epoch)
    # Logging results on tensorboard
    log_value('{}/accuracy1'.format(mode), correct1.avg, step=epoch)
    log_value('{}/accuracy5'.format(mode), correct5.avg, step=epoch)
    return test_loss.avg, correct1.avg, correct5.avg

def validate(args, model, loader, criterion, epoch, mode='val'):
    """
    Evaluates model on validation/test set and logs score on tensorboard.
    """
    test_loss = AverageMeter('Loss', ':.4e')
    correct1 = AverageMeter('Acc@1', ':6.2f')  # for classification
    correct5 = AverageMeter('Acc@5', ':6.2f')  # for classification
    # switch to evaluate mode
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i, (data, target, *_) in enumerate(loader):
            data, target = {k: v.to(args.device) for k, v in data.items()}, target.to(args.device)
            output = model(**data)['logits']
            loss = criterion(output, target)
            loss = loss.mean()
            # measure accuracy and record loss
            test_loss.update(loss.item(), target.size(0))
            acc1, acc5 = compute_topk_accuracy(output, target, topk=(1, 5))
            correct1.update(acc1[0].item(), target.size(0))
            correct5.update(acc5[0].item(), target.size(0))
    log(args.logpath, 'Time for {}-Epoch-{}/{}:{:.1f}s Acc1:{}, Acc5:{}, Loss:{}\n'.format(mode.capitalize(),
                                                                                      # 'Test'if mode=='test'else 'Val',
                                                                                      epoch, args.epochs,
                                                                                      time.time() - t0,
                                                                                      correct1.avg, correct5.avg,
                                                                                      test_loss.avg))
    log_value('{}/loss'.format(mode), test_loss.avg, step=epoch)
    # Logging results on tensorboard
    log_value('{}/accuracy1'.format(mode), correct1.avg, step=epoch)
    log_value('{}/accuracy5'.format(mode), correct5.avg, step=epoch)
    return test_loss.avg, correct1.avg, correct5.avg