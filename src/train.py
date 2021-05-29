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
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_dataloader import TripletImageLoader
from triplet_network import Tripletnet
from networks import *
from visdom import Visdom
import numpy as np
import pdb

print ("Import Successful")

# Training Settings

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
parser.add_argument('--name', default='TripletNet', type=str, help='name of experiment')

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
    train_loader = torch.utils.data.DataLoader(TripletImageLoader(base_path='/scratch', filenames_filename='../data/bam_filename.txt', triplets_filename='../data/bam_training_triplet_filename.txt', transform=transforms.Compose([transforms.ToTensor()])), batch_size = args.batch_size, shuffle=True, **kwargs)
    
    #testing_loader - Remember to update filenames_filename, triplet_filename
    test_loader = torch.utils.data.DataLoader(TripletImageLoader(base_path='/scratch', filenames_filename='../data/bam_filename.txt', triplets_filename='../data/bam_testing_triplet_filename.txt', transform=transforms.Compose([transforms.ToTensor()])), batch_size = args.batch_size, shuffle=False, **kwargs)

    #model is the embedding network architecture
    model = SomeNet()


    tnet = Tripletnet(model)
    print(tnet)
    
    #Load the Model weights from Stage 1 - which is Softmax Classification,
    #Uncomment the below line and update the path for the model state_dict().

    #checkpoint = torch.load('./runs/TripletNet/checkpoint.pth.tar')

    if args.cuda:
        tnet.cuda()

    print ("Model Initialized")

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print(' + Number of params: {}'.format(n_parameters))
    for epoch in range(1, args.epochs+1):
        # train for 1 epoch
        train(train_loader, tnet, criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(test_loader, tnet, criterion, epoch)

        #remember best_acc and save checkpoint

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if is_best:
            save_checkpoint({'epoch':epoch, 'state_dict':tnet.state_dict(), 'best_prec1':best_acc,}, is_best)
    
    #Extract Embeddings function is called.
    #extract_embeddings(test_loader, model)

def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # Switch to train mode
    tnet.train()

    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)


        # Compute Output
        dist_a, dist_b, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = accuracy(dist_a, dist_b)
        losses.update(loss_triplet.data, data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data/3, data1.size(0))

        #compute gradient and do optimizer step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print ('Train Epoch: {} [{}/{}]\t Loss: {:.4f} ({:.4f}) \t Acc: {:.2f}% ({:.2f}) \t Emb_norm: {:.2f} ({:.2f})'.format(epoch, batch_idx*len(data1), len(train_loader.dataset), losses.val, losses.avg, 100.*accs.val, 100.*accs.avg, emb_norms.val, emb_norms.avg))

    # log avg values to somewhere
    '''
    plotter.plot('acc', 'train', epoch, accs.avg)
    plotter.plot('loss', 'train', epoch, losses.avg)
    plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)
    '''

def test(test_loader, tnet, criterion, epoch):

    losses = AverageMeter()
    accs = AverageMeter()

    #switch to evaluation mode

    tnet.eval()

    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()

        #compute output

        dist_a, dist_b, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dist_a.size()).fill_(1)

        if args.cuda:
            target = target.cuda()

        test_loss = criterion(dist_a, dist_b, target).data

        # measure accuracy
        acc = accuracy(dist_a, dist_b)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(losses.avg, 100. * accs.avg))

    #plotter.plot('acc', 'test', epoch, accs.avg)
    #plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg

def extract_embeddings(some_loader, embedding_model):

    print ("Extracting Embeddings as we speak")

    with torch.no_grad():
        embedding_model.eval()

        out_list = []

        for batch_idx, (data1, data2, data3) in enumerate(some_loader):
            if args.cuda:
                data1 = data1.cuda()

            out = embedding_model.forward(data1).cpu().numpy()
            for i in range(out.shape[0]):
                out_list.append(out[i,:])
        
        out_list = np.asarray(out_list)
        print ("out dimensions", out_list.shape)
        np.savetxt('bottle_neck_bam_triplet_stage_2.txt', out_list)

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
