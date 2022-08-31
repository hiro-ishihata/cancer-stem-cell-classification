# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import CSCDataset, get_transform

def get_model(model_name, n_outputs, pretrained=True):
    """ get n_outputs CNN model 
    Args:
        model_name(str) : 'VGG16' and 'ResNet50'
        n_outputs : number of classes
        pretrained (bool,optional) : initialize using pretrained parameter
    Returns:
        model
    """
    if model_name == 'VGG16':
        if pretrained: # update only final stage 
            model = models.vgg16_bn(pretrained=True)
            model.classifier[6] = torch.nn.Linear(4096, n_outputs)
            params_to_update = []
            for name, param in model.named_parameters():
                if name in ["classifier.6.weight", "classifier.6.bias"]:
                    param.requires_grad = True
                    params_to_update.append(param)
                else:
                    param.requires_grad = False
        else:
            model = models.vgg16_bn(pretrained=False)
            model.classifier[6] = torch.nn.Linear(4096, n_outputs)
    elif model_name == 'ResNet50': 
        if pretrained: # update only final stage 
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(2048, n_outputs)
            params_to_update = []
            for name, param in model.named_parameters():
                if name in ["fc.weight", "fc.bias"]:
                    param.requires_grad = True
                    params_to_update.append(param)
                else:
                    param.requires_grad = False
        else:
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(2048, n_outputs)
    return model
    

def drawgraph(train_acc, train_loss, test_acc, test_loss, fname): 
    """ draw accuracy and loss graph
    Args:
        train_acc (List(float)): accuracy of training data
        train_loss (List(float)): loss of training data
        test_acc (List(float)): accuracy of testing data
        test_loss (List(float)): loss of testing data
        fname (str): file name of figure
    Returns:
        fig
    """
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].plot(range(len(train_loss)), train_loss, label='train')
    ax[0].plot(range(len(test_loss)), test_loss, label='test')
    ax[1].plot(range(len(train_acc)), train_acc, label='train')
    ax[1].plot(range(len(test_acc)), test_acc, label='test')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title("Loss")
    ax[1].set_title("Accuracy")
    fig.savefig(fname)
    return fig


def accuracy(y, t):
    """ calculate accuracy 
    Args:
        y (Tensor): predict data
        t (Tensor): ground truth data
    Returns:
        accuracy
    """
    pred = y.argmax(axis=1).reshape(t.shape)
    return ((pred == t).sum()/float(len(t))).item()


def do_epoch(dataloader, model, opt, criterion, device):
    """ one epoch training function 
    Args:
        dataloader: dataloader object
        model: CNN model
        opt: optimizer
        criterion: function to calculate loss
        device: cuda/cpu
    Returns:
        loss and accuracy
    """
    if opt: model.train()  # training mode if optimizer is not None
    else:   model.eval()   # evaluation mode 
    sum_acc, sum_loss = 0, 0 # accuracy and loss for 1 epoch
    for x, t in dataloader:       # forward and backprop
        x = x.to(device)
        t = t.to(device)
        res = model(x)            # forward calculation
        loss = criterion(res, t)  # loss calculation
        acc = accuracy(res, t)
        if opt:
            opt.zero_grad()       # initialize grad
            loss.backward()       # back propagation
            opt.step()
        sum_loss += loss.item()   # accumulate accuracy and loss
        sum_acc += acc
    return sum_loss/len(dataloader), sum_acc/len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='csc train')
    parser.add_argument('--batchsize', type=int, default=64, help='Minibatch size')
    parser.add_argument('--outd', type=str, default='result', help='Output directory')
    parser.add_argument('--epoch', type=int, default=30, help='Number of epoch')
    parser.add_argument('--model', type=str, default='VGG16', help='CNN model')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    args = parser.parse_args()
    if not os.path.exists(args.outd):
        os.mkdir(args.outd)

    train_d = CSCDataset(transfunc=get_transform('train'), 
                         seed=123, folds=[0,1,2,3])
    test_d = CSCDataset(transfunc=get_transform('test'), 
                        seed=123, folds=[4])
    train_dl = DataLoader(train_d, 
                          batch_size=args.batchsize, shuffle=True)
    test_dl = DataLoader(test_d, 
                         batch_size=args.batchsize, shuffle=False)
    print(f"data size train={len(train_d)} test={len(test_d)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_outputs = 2
    model = get_model(args.model, n_outputs, args.pretrained) 
    model = model.to(device)
    opt = optim.SGD(model.parameters(), lr=0.01,momentum=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []

    for epoch in range(1, args.epoch+1): # training loop
        print ('epoch', epoch, flush=True)
        sum_loss, sum_acc = do_epoch(train_dl, model, opt, criterion, device)
        print (f'train mean loss={sum_loss}, accuracy={sum_acc}')
        train_loss.append(sum_loss)
        train_acc.append(sum_acc)

        with torch.no_grad(): # evaluation
            sum_loss, sum_acc = do_epoch(test_dl, model, None, criterion, device)
        print (f'test mean loss={sum_loss}, accuracy={sum_acc}')
        test_loss.append(sum_loss)
        test_acc.append(sum_acc)
        
    fname = args.outd + '/' + args.model+ ('_pre' if args.pretrained else '')
    drawgraph(train_acc, train_loss, test_acc, test_loss, fname + "_train.png")
    torch.save(model.state_dict(), fname+ "_model.pt")

