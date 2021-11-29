# REPVGG/utils.py

import torch
import math
import os
import random
import torchvision.transforms as tfs
import PIL

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        okay = ((target - pred)<=3).sum()
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

def log_msg(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        print(message, file=f)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform, args, valid=False):
        self.file_path = file_path
        self.resize = args.resize
        files = os.listdir(file_path)
        random.shuffle(files)
        if valid:
            self.file_list = [file_path+'/'+f for f in files[10001:]]
        else:
            self.file_list = [file_path+'/'+f for f in files[:10000]]
        self.transform = transform
        self.valid = valid
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        image = self.file_list[index]
        image = PIL.Image.open(image).convert('RGB').resize((self.resize, self.resize))
        angle = random.randint(0, 359)
        image = image.rotate(angle)
        image = self.transform(image)
        return image, angle

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform, args):
        self.file_list = [file_path+f for f in os.listdir(file_path)] * 360
        self.transform = transform
        self.resize = args.resize
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        image = self.file_list[index]
        angle = index//4
        image = PIL.Image.open(image).convert('RGB').resize((self.resize, self.resize), reducing_gap=3)
        image = image.rotate(angle)
        image = self.transform(image)
        return image, angle

def get_train_dataset(args, trans):
    traindir = os.path.join(args.data, 'train')
    train_dataset = TrainDataset(traindir, trans, args)
    return train_dataset

def get_val_dataset(args, trans):
    traindir = os.path.join(args.data, 'train')
    val_dataset = TrainDataset(traindir, trans, args, valid=True)
    return val_dataset

def get_test_dataset(args, trans):
    testdir = os.path.join(args.testdata, '')
    test_dataset = TestDataset(testdir, trans, args)
    return test_dataset

def get_train_trans(args):
    transform = tfs.Compose([
        tfs.CenterCrop(args.resolution),
        tfs.Grayscale(3),
        tfs.ColorJitter(.4, .4, 0, .4),
        tfs.ToTensor(),
        tfs.Normalize([.5, .5, .5], [.5, .5, .5]),        
        ])
    return transform

def get_val_trans(args):
    transform = tfs.Compose([
        tfs.CenterCrop(args.resolution),
        tfs.ToTensor(),
        tfs.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    return transform

def get_train_loader(args):
    train_trans = get_train_trans(args)
    train_dataset = get_train_dataset(args, train_trans)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    return train_loader

def get_val_loader(args):
    val_trans = get_val_trans(args)
    val_dataset = get_val_dataset(args, val_trans)
    if hasattr(args, 'val_batch_size'):
        bs = args.val_batch_size
    else:
        bs = args.batch_size

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bs, shuffle=False, 
        num_workers=args.workers, pin_memory=True)
    return val_loader

def get_test_loader(args):
    test_trans = get_val_trans(args)
    test_dataset = get_test_dataset(args, test_trans)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return test_loader