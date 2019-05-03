# -*- coding: utf-8 -*-
"""
=========================
Qian Wang, Ning Jia, Toby P. Breckon, A Baseline for Multi-Label Image Classification Using Ensemble Deep CNN, ICIP 2019
**Author**: Qian Wang
qian.wang173@hotmail.com
==========================

"""
# License: BSD
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from evaluation_metrics import prf_cal, cemap_cal
import scipy
import scipy.io
from myvgg import vgg16
import pdb


trial = 2
img_size = 448
if len(sys.argv)>=2:
    trial = int(sys.argv[1])
if len(sys.argv)>=3:
    img_size = int(sys.argv[2])
randseed = trial
np.random.seed(randseed)
torch.manual_seed(randseed)

class VocDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        data_vec = self.labels.iloc[idx, :].values
        img_name = '%06d'%data_vec[0]+'.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        #label = np.array(data_vec[1:]>-1, dtype=int)
        label = np.array(data_vec[1:],dtype=int)
        label = label.astype('double')
# =============================================================================
#         if len(image.shape)==2:
#             image = np.expand_dims(image,2)
#             image = np.concatenate((image,image,image),axis=2)
# =============================================================================
        if not image.mode=='RGB':
            image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

data_transforms = {
    'trainval': transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomResizedCrop((448)),
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomResizedCrop((224)),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# =============================================================================
# image_datasets = {x: VocDataset('~/Qian/labels/'+'classification_'+x+'.csv',
#                                  '~/Qian/VOCdevkit/VOC2007/JPEGImages/',
#                                  data_transforms[x])
#                   for x in ['train', 'val']}
# =============================================================================
image_datasets = {x: VocDataset('./'+x+'.csv',
                                 './VOCdevkit/VOC2007/JPEGImages/', data_transforms[x])
                  for x in ['trainval', 'val']}
bs=16
dataloaders = {'train': 0, 'val': 0}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['trainval'], batch_size=bs,shuffle=True, num_workers=4)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=bs,shuffle=False, num_workers=4)
dataset_sizes = {x: len(image_datasets[x]) for x in ['trainval', 'val']}

image_datasets_test = VocDataset('./classification_test.csv', 
                                 './VOCdevkit/VOC2007/JPEGImages/', data_transforms['test'])
dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size=bs, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######################################################################
# Mixup data for data augmentation
# ^^^^^^^^^^^^^^^^^^^^^^
# https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = (y + y[index,:])>0
    mixed_y = mixed_y.float()
    return torch.cat((x,mixed_x),0), torch.cat((y,mixed_y),0)

def mixup_data2(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = y
    return torch.cat((x,mixed_x),0), torch.cat((y,y),0)

def mixup_data3(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = (y + y[index,:])>0
    mixed_y = mixed_y.float()
    return mixed_x, mixed_y


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_emap = 0
    for epoch in range(num_epochs):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('Current learning rate: ' + '%.5f'%cur_lr)
        if cur_lr < 0.00001:
            break      
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # mixed up
                inputs = inputs.float()
                labels = labels.float()
                labels_tr = labels.clone()
                labels_tr[labels==0] = 1
                labels_tr[labels==-1] = 0
                #if epoch % 2 == 0:
                #    inputs,labels_tr = mixup_data3(inputs,labels_tr,use_cuda=True)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_tr)
                    # loss = newHinge_rank_loss(outputs,labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        if index == 0:
                            outputs_val = outputs
                            labels_val = labels
                        else:
                            outputs_val = torch.cat((outputs_val,outputs),0)
                            labels_val = torch.cat((labels_val,labels),0)
                running_loss += loss.item() * inputs.size(0)
            if phase=='train':
                epoch_loss = running_loss / dataset_sizes['trainval']
            else:
                epoch_loss = running_loss / dataset_sizes['val']
            if phase == 'val':
                cmap,emap = cemap_cal(outputs_val.to(torch.device("cpu")).numpy(),labels_val.to(torch.device("cpu")).numpy())
                p,r,f = prf_cal(outputs_val.to(torch.device("cpu")).numpy(),labels_val.to(torch.device("cpu")).numpy(),3)
                val_loss = epoch_loss
                
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and cmap > best_emap:
                best_emap = cmap
                print(cmap,emap)
                print(p,r,f)
            test_model(model,optimizer,trial)
            #best_model_wts = copy.deepcopy(model.state_dict())
            #torch.save(model_ft.state_dict(),'./vgg16_model1_trial0.pt')
            torch.save(model_ft.state_dict(),'./vgg16_model1fc_trial'+str(trial)+'imgSize'+str(img_size)+'.pt')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model,optimizer,trial):
    since = time.time()
    model.eval()
    # Iterate over data.
    for index, (inputs, labels) in enumerate(dataloaders_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            # backward + optimize only if in training phase
            if index == 0:
                outputs_test = outputs
                labels_test = labels
            else:
                outputs_test = torch.cat((outputs_test, outputs), 0)
                labels_test = torch.cat((labels_test, labels), 0)

    cmap, emap = cemap_cal(outputs_test.to(torch.device("cpu")).numpy(), labels_test.to(torch.device("cpu")).numpy())
    print('Test:')
    print(cmap,emap)
    p, r, f = prf_cal(outputs_test.to(torch.device("cpu")).numpy(), labels_test.to(torch.device("cpu")).numpy(), 3)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    scipy.io.savemat('./results/vgg16_model1_results_'+str(trial)+'imgSize'+str(img_size)+'.mat', mdict={'cmap': cmap, 'emap': emap, 'p': p,'r': r, 'f': f, 'scores': outputs_test.to(torch.device("cpu")).numpy()})
    # load best model weights
    # return cmap,emap,p,r,f,outputs_test.to(torch.device("cpu")).numpy()


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model.
#


#model_ft = models.vgg16(pretrained=True)
#for param in model_ft.parameters():
#    param.requires_grad = False
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 20)
num_classes=20
model_ft = vgg16(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features
model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
model_ft = model_ft.to(device)

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MultiLabelSoftMarginLoss()
# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_ft = optim.SGD([{'params':model_ft.features.parameters(),'lr':1e-2},{'params': model_ft.classifier.parameters()}],lr=0.1, momentum=0.9)
# optimizer_ft = optim.Adam([{'params': model_ft.layer5.parameters()},{'params':model_ft.features.parameters(),'lr':1e-5}], lr=0.0001)
# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min',factor=0.1,patience=5)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
#####################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
if trial>10:
    model_ft.load_state_dict(torch.load('./results/vgg16_model1_trial0.pt'))
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=100)
else:
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=40)
torch.save(model_ft.state_dict(),'./results/vgg16_model1_trial'+str(trial)+'imgSize'+str(img_size)+'.pt')
# model_ft.load_state_dict(torch.load('./vgg16_model_test.pt'))
######################################################################
#

