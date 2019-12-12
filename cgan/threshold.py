# from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import vgg


from tqdm import trange, tqdm
from datetime import datetime


from datasets import CombinedDataset, ThresholdedDataset


def print_log(s, save_dir):
    print(s)
    with open(os.path.join(save_dir, "log.txt"), "a+") as f:
        f.write(str(s) + "\n")


def main():
    model_names = sorted(name for name in vgg.__dict__ if name.islower() and not name.startswith("__") and name.startswith("vgg") and callable(vgg.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',  choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=25, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu')
    parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models', default=os.path.join('run_logs', datetime.now().strftime("%m_%d_%H_%M_%S")) + "__threshold", type=str)

    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--save-freq', default=5, type=int, metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--mode', default="train", type=str, metavar='MODE', help='mode: [train, evaluate, threshold]')



    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print_log(args, args.save_dir)

    model = vgg.__dict__[args.arch](n_classes=2)

    model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    if args.mode == "threshold":
        args.resume = "../vggnet/run_logs/12_02_01_43_33/checkpoint_20.tar"
        assert os.path.isfile(args.resume)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.mode != "threshold":
        # create train dataloader without normalization to find mean/std for normalization
        train_dataloader_without_normalization = torch.utils.data.DataLoader(
            CombinedDataset(
                p_cifar=1.0, 
                p_thresholded=1.0,
                threshold=0.0,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                train=True,
            ),
            batch_size=args.batch_size, 
            num_workers=args.workers,
        )
        mean, std = get_mean_std(train_dataloader_without_normalization)
        normalize = transforms.Normalize(mean=mean, std=std)

        if args.augment:
            print("AUGMENTING")
            transformations = [
                lambda x: x.cpu(),
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                lambda x: x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                normalize,
            ]
        else:
            print("NOT AUGMENTING")
            transformations = [normalize]

        train_loader = torch.utils.data.DataLoader(
            CombinedDataset(
                p_cifar=1.,
                p_thresholded=1.,
                threshold=0.0,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                transform=transforms.Compose(transformations),
                p_val=0.1,
                train=True,
            ),
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.workers,
        )

        val_loader = torch.utils.data.DataLoader(
            CombinedDataset(
                p_cifar=1., 
                p_thresholded=1.,
                threshold=0.0,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                transform=transforms.Compose(transformations),
                p_val=0.1,
                train=False,
            ),
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.workers, 
        )
    else:
        threshold_loader = torch.utils.data.DataLoader(
            ThresholdedDataset(0.),
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.workers, 
        )
        single_loader = torch.utils.data.DataLoader(
            ThresholdedDataset(0.),
            batch_size=1, 
            shuffle=False,
            num_workers=args.workers, 
        )

    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss()
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

    if args.mode == "evaluate":
        validate(val_loader, model, criterion)

    elif args.mode == "train":
        for epoch in trange(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args.epochs)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if epoch % args.save_freq == 0 or epoch == (args.epochs - 1):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

    elif args.mode == "threshold":
        threshold(model, threshold_loader, single_loader)

    else:
        raise Exception("Unrecognized mode " + args.mode)

def threshold(model, threshold_loader, single_loader):
    realness = None
    model.eval()
    for i, (input, _) in enumerate(threshold_loader):
        if args.cpu == False:
            input = input.cuda()
        if args.half:
            input = input.half()

        # compute output
        output = model(input).squeeze(dim=-1).detach()
        if realness is None:
            realness = output
        else:
            realness = torch.cat((realness, output))

    # thresh = 0.5
    # print(realness)
    # print(realness.max(), realness.min(), realness.median())
    # print((realness >= thresh).float().sum(), (realness >= thresh).float().mean())

    os.makedirs("thresholded_images/", exist_ok=True)
    for topk in tqdm([196, 1000, 5000, 10000, 25000]): # 196 of them are >= 0.5
        indices = [index.item() for index in realness.topk(k=topk)[1]]
        assert len(indices) == topk
        data = None
        labels = None
        for i, (input, target) in enumerate(single_loader):
            if i not in indices:
                continue
            if args.cpu == False:
                input = input.cuda()
            if args.half:
                input = input.half()

            if data is None:
                data = input
                labels = target
            else:
                data = torch.cat((data, input))
                labels = torch.cat((labels, target))

        assert data is not None
        assert len(data) == topk
        torch.save((data, labels), f"thresholded_images/top_{topk}.tar")


def train(train_loader, model, criterion, optimizer, epoch, max_epochs):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, _, target) in enumerate(train_loader):
        target = target.float().unsqueeze(dim=1)

        # measure data loading time
        data_time.update(time.time() - end)

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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch + 1, max_epochs, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1), args.save_dir)


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, _, target) in enumerate(val_loader):
        target = target.float().unsqueeze(dim=1)
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1), args.save_dir)

    print_log(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1), args.save_dir)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
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
    """Computes the precision@k for the specified values of k"""
    return [(output.round() == target).float().mean() * 100]
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
    for data, _ ,_ in tqdm(loader):
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
