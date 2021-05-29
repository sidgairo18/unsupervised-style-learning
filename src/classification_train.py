# Code adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/train.py
from __future__ import print_function, division
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from classification_dataloader import ClassificationImageLoader
from networks import *
from visdom import Visdom
import numpy as np
import pdb

print ("Import Successful")

# Training Settings

parser = argparse.ArgumentParser(description='PyTorch MNIST Example Classification')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch_size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='helps enable cuda training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default:1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M', help='margin for triplet loss (default: 2)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint, default: None')
parser.add_argument('--n-classes', default=11, type=int, help='Number of classes for classification')
parser.add_argument('--name', default='Classification Net', type=str, help='name of experiment')

print ("Training Settings updated")

best_acc = 0

def main():

    global args, best_acc
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    global plotter
    #plotter = VisdomLinePlotter(env_name=args.name)

    kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}

    #training loader
    train_loader = torch.utils.data.DataLoader(ClassificationImageLoader(base_path='../data', filenames_filename='../data/bam_classification_training_filename.txt', labels_filename='../data/bam_classification_training_labels.txt', transform=transforms.Compose([transforms.ToTensor()])), batch_size = args.batch_size, shuffle=True, **kwargs)
    
    #testing_loader - Remember to update filenames_filename, triplet_filename
    test_loader = torch.utils.data.DataLoader(ClassificationImageLoader(base_path='../data', filenames_filename='../data/bam_classification_testing_filename.txt', labels_filename='../data/bam_classification_testing_labels.txt', transform=transforms.Compose([transforms.ToTensor()])), batch_size = args.batch_size, shuffle=False, **kwargs)
    
    #model is the embedding network architecture
    model = SomeNet()

    #Classification Network
    class_net = ClassificationNet(model, n_classes = args.n_classes)

    print ("Number of classes", args.n_classes)
    print (class_net)

    if args.cuda:
        class_net.cuda()

    print ("Model Initialized")

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            class_net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(class_net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    n_parameters = sum([p.data.nelement() for p in class_net.parameters()])
    print(' + Number of params: {}'.format(n_parameters))

    for epoch in range(1, args.epochs+1):

        scheduler.step()

        # train for 1 epoch
        train_loss, metrics = train(train_loader, class_net, criterion, optimizer, scheduler, epoch, metrics=[AccumulatedAccuracyMetric()])
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, args.epochs+1, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        
        # evaluate on validation set
        val_loss, metrics = test(test_loader, class_net, criterion, epoch, metrics=[AccumulatedAccuracyMetric()])
        val_loss /= len(test_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, args.epochs+1, val_loss)
        
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print (message)

        
    #extract_embeddings(test_loader, model)

def train(train_loader, class_net, criterion, optimizer, scheduler, epoch, metrics):

    for metric in metrics:
        metric.reset()

    # Switch to train mode
    class_net.train()
    losses = []
    total_loss = 0.0

    for batch_idx, (data1, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data1) in (tuple, list):
            data1 = (data1, )
        if args.cuda:
            data1 = tuple(d.cuda() for d in data1)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        
        # Compute Output
        outputs = class_net(*data1)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs

        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = criterion(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % args.log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data1[0]), len(train_loader.dataset), 100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print (message)
            losses = []


    total_loss /= (batch_idx+1)
    return total_loss, metrics
    # log avg values to somewhere
    '''
    plotter.plot('acc', 'train', epoch, accs.avg)
    plotter.plot('loss', 'train', epoch, losses.avg)
    plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)
    '''

def test(test_loader, class_net, criterion, epoch, metrics):

    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        #switch to evaluation mode

        class_net.eval()
        val_loss = 0.0

        for batch_idx, (data1, target) in enumerate(test_loader):
            target = target if len(target) > 0 else None
            if not type(data1) in (tuple, list):
                data1 = (data1, )
            if args.cuda:
                data1 = tuple(d.cuda() for d in data1)
                if target is not None:
                    target = target.cuda()


            #compute output
            outputs = class_net(*data1)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs

            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = criterion(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()
            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

def extract_embeddings(some_loader, embedding_model):

    print ("Extracting Embeddings as we speak")

    with torch.no_grad():
        embedding_model.eval()

        out_list = []

        for batch_idx, (data1, target) in enumerate(some_loader):
            if args.cuda:
                data1 = data1.cuda()

            out = embedding_model.forward(data1).cpu().numpy()
            for i in range(out.shape[0]):
                out_list.append(out[i,:])
        
        out_list = np.asarray(out_list)
        print ("out dimensions", out_list.shape)
        np.savetxt('classification_bam_features_alex_30.txt', out_list)

def accuracy(dist_a, dist_b):
    margin = 0
    pred = (dist_a - dist_b - margin).cpu().data
    return float((pred > 0).sum()*1.0)/(dist_a.size()[0])

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')


###################### Class AverageMeter ################################

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

###################### Class AverageMeter ################################

###################### Class Metric ################################

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

###################### Class Metric ################################


###################### Class Accumulated Accuracy Metrics ################################

class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


###################### Class Accumulated Accuracy Metrics ################################




##################### Class for VisdomLinePlotter ##########################
class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)
##################### Class for VisdomLinePlotter ##########################



if __name__ == "__main__":
    main()
