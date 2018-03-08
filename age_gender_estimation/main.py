import matplotlib.pyplot as plt

from skimage.io import imshow
import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import time
import shutil

# Transform this later into command line arguments
arg_lr = 1e-6
epochs = 1000
print_freq = 10
n_processors = 4
batch_size = 128

transforms_train = tv.transforms.Compose([
    tv.transforms.Resize(314),
    tv.transforms.CenterCrop(224),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomGrayscale(),
    tv.transforms.ColorJitter(),
    tv.transforms.RandomRotation(30),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_test_eval = tv.transforms.Compose([
    tv.transforms.Resize(280),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = tv.datasets.ImageFolder('data/train/', transform=transforms_train)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_process
ors, pin_memory=True, drop_last=True)

dataset_eval = tv.datasets.ImageFolder('data/eval/', transform=transforms_test_eval)
dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=n_processor
s, pin_memory=True, drop_last=True)

dataset_test = tv.datasets.ImageFolder('data/test/', transform=transforms_test_eval)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=n_processor
s, pin_memory=True, drop_last=True)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch+1) % 30 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (0.1 ** (epoch // 30))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg
    
# define model

model = tv.models.resnet101(pretrained=True)
model.fc = nn.Sequential(
               nn.Linear(2048, 1024, bias=False),
               nn.SELU(inplace=True),
               nn.Dropout(),
               nn.Linear(1024, 512, bias=False),
               nn.SELU(inplace=True),
               nn.Dropout(),
               nn.Linear(512, 36)
)

model = torch.nn.DataParallel(model).cuda()

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.Adam([
    {'params': model.module.conv1.parameters()},
    {'params': model.module.bn1.parameters()},
    {'params': model.module.relu.parameters()},
    {'params': model.module.maxpool.parameters()},
    {'params': model.module.layer1.parameters()},
    {'params': model.module.layer2.parameters()},
    {'params': model.module.layer3.parameters()},
    {'params': model.module.layer4.parameters()},
    {'params': model.module.avgpool.parameters()},
    {'params': model.module.fc.parameters(), 'lr': arg_lr*100}
], arg_lr)

best_prec1 = 0

for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(dataloader_train, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(dataloader_eval, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
