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
import torchvision.transforms as tfs
import PIL
import copy
from utils import accuracy, ProgressMeter, AverageMeter
from model import get_RepVGG_func_by_name

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--testdata', default='../data/test_data/')
parser.add_argument('--weights', default='result/RepVGG-A0se_norm_test_best.pth.tar', help='path to the weights file')
parser.add_argument('--mode', default='train', choices=['train', 'deploy'])
parser.add_argument('--arch', metavar='ARCH', default='RepVGG-A0se')
parser.add_argument('--workers', default=4, type=int, metavar='N')
parser.add_argument('--batch-size', default=1)
parser.add_argument('--resize', default=320)
parser.add_argument('--resolution', default=128)
parser.add_argument('--tag', default='convert_model')

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'swith_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model

def test():
    args = parser.parse_args()

    repvgg_build_func = get_RepVGG_func_by_name(args.arch)
    model = repvgg_build_func(deploy=False)
    model.cuda()

    if os.path.isfile(args.weights):
        print('=> loading checkpoint "{}"'.format(args.weights))
        load_checkpoint(model, args.weights)
    cudnn.benchmark=True
    test_loader = get_test_loader(args)
    acc, label, predict = validate(test_loader, model)
    save = {'label' : label, 'predict':predict}
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


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform, args):
        self.file_list = [file_path+f for f in os.listdir(file_path)]
        self.transform = transform
        self.resize = args.resize
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        image = self.file_list[index]
        angle = index//4
        image = PIL.Image.open(image).convert('RGB').resize((self.resize, self.resize))
        image = image.rotate(angle)
        image = self.transform(image)
        return image, angle


def get_test_dataset(args, trans):
    testdir = os.path.join(args.testdata, '')
    test_dataset = TestDataset(testdir, trans, args)
    return test_dataset

def get_test_loader(args):
    test_trans = get_val_trans(args)
    test_dataset = get_test_dataset(args, test_trans)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return test_loader
    
def get_val_trans(args):
    transform = tfs.Compose([
        tfs.CenterCrop(args.resolution),
        tfs.ToTensor(),
        tfs.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    return transform

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            ckpt[k[7:]] = v
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt)

if __name__ == '__main__':
    test()