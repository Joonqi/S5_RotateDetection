# REPVGG/train.py

import argparse
import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import AverageMeter, accuracy, ProgressMeter, \
        get_val_loader, get_train_loader, get_test_loader, log_msg
from torchsummary import summary

from model import get_RepVGG_func_by_name

parser = argparse.ArgumentParser(description='PyTorch RepVGG Train')
parser.add_argument('--data', default='../data', metavar='DIR')
parser.add_argument('--testdata', default='../data/test_samples/easyset/')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0se')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')
parser.add_argument('--epochs', default=120, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--val-batch-size', default=32, type=int, metavar='V')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float, dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int)
parser.add_argument('--resize', default=320)
parser.add_argument('--resolution', default=128)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--world-size', default=-1, type=int)
parser.add_argument('--rank', default=-1, type=int)
parser.add_argument('--seed', default = None, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--multiprocessing-distributed', action='store_true')
parser.add_argument('--custwd', dest='custwd', action='store_true')
parser.add_argument('--tag', default='drop_dropout', type=str)
TRAINSET_SIZE = 10000
best_acc1 = 0

def adam_optimizer(model, lr, weight_decay, use_custwd):
    params = []
    for key, value in model.named_parameters():
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1 in key')) or 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
        params += [{'params':[value], 'lr':apply_lr, 'weight_decay':apply_weight_decay}]
    optimizer = torch.optim.Adam(params, lr)
    return optimizer

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    log_file = './result/log_{}_{}_train.txt'.format(args.arch, args.tag)
    repvgg_build_func = get_RepVGG_func_by_name(args.arch)
    model = repvgg_build_func(deploy=False)
    
    is_main = not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node==0)
    
    if is_main:
        for n, p in model.named_parameters():
            print(n, p.size())
        for n, p in model.named_buffers():
            print(n, p.size())
        log_msg('epochs {}, lr {}, weight_decay {}, tag {}'.format(args.epochs, args.lr, args.weight_decay, args.tag), log_file)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = adam_optimizer(model, args.lr, args.weight_decay, args.custwd)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint "{}"'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    summary(model, (3, args.resolution, args.resolution))

    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    for epoch in range(args.start_epoch, args.epochs):
        if epoch % 5 == 0:
            test_loader = get_test_loader(args)
            acc1, loss = validate(test_loader, model, criterion, args)
            msg = 'Test data: {}, acc {:.4f}, Loss {:.4f}'.format(epoch, acc1, loss)
            log_msg(msg, log_file)

        train(train_loader, model, criterion, optimizer, epoch, args, is_main=is_main)
        if is_main:
            acc1, loss = validate(val_loader, model, criterion, args)
            msg = '{}, epoch {}, acc {:.4f}, Loss {:.4f}'.format(args.arch, epoch, acc1, loss)
            log_msg(msg, log_file)
            
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({
                'epoch':epoch+1,
                'arch': args.arch,
                'state_dict':model.state_dict(),
                'best_acc1':best_acc1,
                'optimizer':optimizer.state_dict(),
            }, is_best, filename='./result/{}_{}.pth.tar'.format(args.arch, args.tag),
            best_filename = './result/{}_{}_best.pth.tar'.format(args.arch, args.tag))

        
def train(train_loader, model, criterion, optimizer, epoch, args, is_main):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5, ],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()

        if is_main and i% args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Loss {loss:3.3f} Acc@1 {top1.avg:3.3f} Acc@5 {top5.avg:.3f}'.format(loss=loss, top1=top1, top5=top5))
    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    main()