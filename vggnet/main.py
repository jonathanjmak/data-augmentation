# from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py

import argparse
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg


from tqdm import trange, tqdm
from datetime import datetime
from pprint import pformat

from datasets import CombinedDataset


def log(s, save_dir):
    with open(os.path.join(save_dir, "log.txt"), "a+") as f:
        f.write(str(s) + "\n")

def print_log(s, save_dir):
    print(s)
    log(s, save_dir)
    


def main():
    model_names = sorted(name for name in vgg.__dict__ if name.islower() and not name.startswith("__") and name.startswith("vgg") and callable(vgg.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',  choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu')

    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--instagram', action='store_true')
    parser.add_argument('--p_cifar', default=1.0, type=float, help='proportion of cifar dataset to use')
    parser.add_argument('--p_thresholded', default=0.0, type=float, help='proportion of synthetic (gan generated) dataset to use')
    parser.add_argument('--p_monet', default=0.0, type=float, help='proportion of monet stylized dataset to use')
    parser.add_argument('--p_udnie', default=0.0, type=float, help='proportion of udnie stylized dataset to use')
    parser.add_argument('--threshold', default=None, type=float, help='realness of synthetic dataset')
    parser.add_argument('--save-freq', default=20, type=int, metavar='N', help='print frequency (default: 20)')



    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    args.save_dir = os.path.join('run_logs', datetime.now().strftime("%m_%d_%H_%M_%S") + "__" + "__".join([arg.replace("--", "") for arg in sys.argv[1:]]))

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print_log(pformat(vars(args)), args.save_dir)

    model = vgg.__dict__[args.arch]()

    model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create train dataloader without normalization to find mean/std for normalization
    train_dataloader_without_normalization = torch.utils.data.DataLoader(
        CombinedDataset(
            p_cifar=args.p_cifar, 
            p_thresholded=args.p_thresholded,
            threshold=args.threshold,
            p_monet=args.p_monet,
            p_udnie=args.p_udnie,
            device=device,
            train=True,
        ),
        batch_size=args.batch_size, 
        num_workers=args.workers,
    )
    mean, std = get_mean_std(train_dataloader_without_normalization)
    print(f"mean: {mean}, std: {std}")
    normalize = transforms.Normalize(mean=mean, std=std)

    if args.augment and args.instagram:
        print("AUGMENTING and FILTERING")
        transformations = [
            lambda x: x.cpu(),
            transforms.ToPILImage(),
            transforms.ColorJitter(),
#             transforms.RandomGrayscale(p=0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            lambda x: x.to(device),
            normalize,
        ]
    elif args.augment and not args.instagram:
        print("AUGMENTING ONLY")
        transformations = [
            lambda x: x.cpu(),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            lambda x: x.to(device),
            normalize,
        ]
    elif not args.augment and args.instagram:
        print("FILTERING ONLY")
        transformations = [
            lambda x: x.cpu(),
            transforms.ToPILImage(),
            transforms.ColorJitter(),
#             transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            lambda x: x.to(device),
            normalize,
        ]
    else:
        print("NOT AUGMENTING OR FILTERING")
        transformations = [normalize]
        

    print("Using combined dataset!")
    train_loader = torch.utils.data.DataLoader(
        CombinedDataset(
            p_cifar=args.p_cifar, 
            p_thresholded=args.p_thresholded,
            threshold=args.threshold,
            p_monet=args.p_monet,
            p_udnie=args.p_udnie,
            device=device,
            transform=transforms.Compose(transformations),
            train=True,
        ),
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./data', 
            train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), 
            download=True,
        ),
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
    )

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    global pbar
    with trange(args.start_epoch, args.epochs) as pbar:
        for epoch in pbar:
            try:
                adjust_learning_rate(optimizer, epoch)

                # train for one epoch
                train(train_loader, model, criterion, optimizer, epoch, args.epochs)

                # evaluate on validation set
                prec1 = validate(val_loader, model, criterion)

                # remember best prec@1 and save checkpoint
                if prec1 > best_prec1:
                    best_prec1 =prec1
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    }, filename=os.path.join(args.save_dir, 'best_model.path'.format(epoch)))
            except KeyboardInterrupt:
                if epoch < .9 * args.epochs:
                    shutil.rmdir(args.save_dir)
                raise


def train(train_loader, model, criterion, optimizer, epoch, max_epochs):
    """
        Run one train epoch
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, batch in enumerate(train_loader):
        input, target = batch[0], batch[1]

        if args.cpu == False:
            input = input.cuda()
            target = target.cuda()
        if args.half:
            input = input.half()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    global desc
    desc = f"[{epoch + 1:3d}/{max_epochs:3d}]  train_loss: {losses.avg:.5f}, train_acc: {top1.avg:.5f}"


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.cpu == False:
            input = input.cuda()
            target = target.cuda()

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    global desc
    desc += f",  val_loss: {losses.avg:.5f},  val_acc: {top1.avg:.5f},  best_val_acc: {max(top1.avg, best_prec1):5f}"

    pbar.set_description_str(desc)
    log(desc, args.save_dir)
    print()

    return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_mean_std(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ ,_ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


if __name__ == '__main__':
    main()
