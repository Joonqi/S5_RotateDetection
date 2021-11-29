# REPVGG/predict.py

import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import accuracy, ProgressMeter, AverageMeter
from model import get_RepVGG_func_by_name, repvgg_model_convert
from utils import load_checkpoint, get_test_loader

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--testdata', default='../data/test_samples/easyset/')
parser.add_argument('--weights', default='RepVGG-A0se__best.pth.tar', help='path to the weights file')
parser.add_argument('--mode', default='train', choices=['train', 'deploy'])
parser.add_argument('--arch', metavar='ARCH', default='RepVGG-A0se')
parser.add_argument('--workers', default=4, type=int, metavar='N')
parser.add_argument('--batch-size', default=1)
parser.add_argument('--resize', default=320)
parser.add_argument('--resolution', default=128)
parser.add_argument('--tag', default='85')

def test():
    args = parser.parse_args()

    repvgg_build_func = get_RepVGG_func_by_name(args.arch)
    model = repvgg_build_func(deploy=False)
    model.cuda()

    if os.path.isfile(args.weights):
        print('=> loading checkpoint "{}"'.format(args.weights))
        load_checkpoint(model, args.weights)
    
    model = repvgg_model_convert(model, save_path='./result/RepVGG_A0se_convert_85.pth.tar')
    cudnn.benchmark=True
    test_loader = get_test_loader(args)
    acc, label, predict = validate(test_loader, model)
    save = {'accuracy': acc, 'label' : label, 'predict':predict}
    with open(f'./predicts/pred_{args.arch}_{args.tag}_{int(time.time())}', 'w') as f:
        json.dump(save, f)

def validate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test :')
    model.eval()
    label, predict = [], []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)

            _, predicted = output.max(1)

            label.extend(target.tolist())
            predict.extend(predicted.tolist())
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if i % 100 == 0:
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))
    return top1.avg, label, predict

if __name__ == '__main__':
    test()