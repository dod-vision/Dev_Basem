
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

from skimage.io import imshow
import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import time
import shutil
import math


# In[2]:


arg_lr = 1e-5
epochs = 1000
print_freq = 10
n_processors = 56
batch_size = 2


# In[3]:


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
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_processors, pin_memory=True, drop_last=True)

dataset_eval = tv.datasets.ImageFolder('data/eval/', transform=transforms_test_eval)
dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=n_processors, pin_memory=True, drop_last=True)

dataset_test = tv.datasets.ImageFolder('data/test/', transform=transforms_test_eval)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=n_processors, pin_memory=True, drop_last=True)


# In[4]:


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
    """Sets the learning rate to the initial LR decayed by 5 every power of 2 epochs"""
    if ((epoch & (epoch - 1)) == 0) and epoch != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 5

def accuracy_gender(output, target):
    return sum(output.round().eq(target))/len(output) * 100

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    if type(pred) != torch.cuda.LongTensor:
        pred = pred.round().type(torch.cuda.LongTensor)
        
    if type(target) != torch.cuda.LongTensor:
        target = target.round().type(torch.cuda.LongTensor)
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0][0]


# In[5]:


def train(train_loader, model, criterion_age, criterion_gender, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_age = AverageMeter()
    losses_gender = AverageMeter()
    top1_gender = AverageMeter()
    top1_age = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # create both targets
        target_age = torch.LongTensor(
            [int(dataset_train.classes[k][1]) for k in target]
        )
        target_gender = torch.FloatTensor(
            [1.0 if dataset_train.classes[k][0] == 'M' else 0.0 for k in target]
        )

        target_age = target_age.cuda(async=True)
        target_gender = target_gender.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_age_var = torch.autograd.Variable(target_age)
        target_gender_var = torch.autograd.Variable(target_gender)

        # compute output
        output_age, output_gender = model(input_var)
        loss_age = criterion_age(output_age, target_age_var)
        loss_gender = criterion_gender(output_gender.view(-1), target_gender_var)

        # measure accuracy and record loss
        prec1_age = accuracy(output_age.data, target_age, topk=(1,))
        prec1_gender = accuracy_gender(output_gender.view(-1).data, target_gender)
        losses_age.update(loss_age.data[0], input.size(0))
        losses_gender.update(loss_gender.data[0], input.size(0))
        top1_gender.update(prec1_gender, input.size(0))
        top1_age.update(prec1_age, input.size(0))
        loss_seq = [loss_age, loss_gender]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        torch.autograd.backward(
            loss_seq, 
            [loss_seq[0].data.new(1).fill_(1) for _ in range(len(loss_seq))]
        )
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
                  'Loss_age {lossa.val:.4f} ({lossa.avg:.4f})\t'
                  'Loss_gender {lossg.val:.4f} ({lossg.avg:.4f})\t'
                  'Prec_age {topa.val:.2f} ({topa.avg:.2f})\t'
                  'Prec_gender {topg.val:.2f} ({topg.avg:.2f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lossa=losses_age, lossg=losses_gender,
                   topa=top1_age,
                   topg=top1_gender))


# In[6]:


def validate(val_loader, model, criterion_age, criterion_gender):
    batch_time = AverageMeter()
    losses_age = AverageMeter()
    losses_gender = AverageMeter()
    top1_gender = AverageMeter()
    top1_age = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        
        # create both targets
        target_age = torch.LongTensor(
            [int(dataset_train.classes[k][1]) for k in target]
        )
        target_gender = torch.FloatTensor(
            [1.0 if dataset_train.classes[k][0] == 'M' else 0.0 for k in target]
        )

        target_age = target_age.cuda(async=True)
        target_gender = target_gender.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_age_var = torch.autograd.Variable(target_age, volatile=True)
        target_gender_var = torch.autograd.Variable(target_gender, volatile=True)
        

        # compute output
        output_age, output_gender = model(input_var)
        loss_age = criterion_age(output_age, target_age_var)
        loss_gender = criterion_gender(output_gender, target_gender_var)

        # measure accuracy and record loss
        prec1_age = accuracy(output_age.data, target_age, topk=(1,))
        prec1_gender = accuracy_gender(output_gender.view(-1).data, target_gender)
        losses_age.update(loss_age.data[0], input.size(0))
        losses_gender.update(loss_gender.data[0], input.size(0))
        top1_gender.update(prec1_gender, input.size(0))
        top1_age.update(prec1_age, input.size(0))
        loss_seq = [loss_age, loss_gender]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                  'Loss_age {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_gender {loss_g.val:.4f} ({loss_g.avg:.4f})\t'
                  'Prec_age {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec_gender {top5.val:.2f} ({top5.avg:.2f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses_age,
                   loss_g=losses_gender, top1=top1_age, top5=top1_gender))

    print(' * Prec_age {top1.avg:.3f} Prec_gender {top5.avg:.3f}'
          .format(top1=top1_age, top5=top1_gender))

    return (top1_age.avg + top1_gender.avg) / 2


# In[7]:


# define model      

class AgeGenderModel(nn.Module):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.resnet = tv.models.resnet101(pretrained=True)
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.fc = nn.Dropout(0.2)
        
        self.agenet = nn.Sequential(
               nn.Linear(2048, 4096),
               nn.SELU(inplace=True),
               nn.Dropout(),
               nn.Linear(4096, 512, bias=False),
               nn.SELU(inplace=True),
               nn.Dropout(),
               nn.Linear(512, 18)
        )
        
        self.gendernet = nn.Sequential(
                nn.Linear(2048, 4096),
                nn.SELU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 512, bias=False),
                nn.SELU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, 1),
                nn.Softmax(0)
        )
        
    def forward(self, X):
        out_resnet = self.resnet(X)
        out_age = self.agenet(out_resnet)
        out_gender = self.gendernet(out_resnet)
        
        return out_age, out_gender

model = torch.nn.DataParallel(AgeGenderModel()).cuda()


# In[8]:



# define loss function (criterion) and optimizer
criterion_gender = nn.BCELoss().cuda()
criterion_age = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {'params': model.module.resnet.parameters()},
    {'params': model.module.agenet.parameters(), 'lr': arg_lr*100},
    {'params': model.module.gendernet.parameters(), 'lr': arg_lr*100}
], arg_lr)


# In[9]:


best_prec1 = 0

for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(dataloader_train, model, criterion_age, criterion_gender, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(dataloader_eval, model, criterion_age, criterion_gender)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
