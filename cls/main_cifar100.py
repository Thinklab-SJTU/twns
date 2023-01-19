import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import argparse

import litenet as QUANWNS
import util as U

def ParseArgs():
    parser = argparse.ArgumentParser(description='TWNs Pytorch Example.')
    parser.add_argument('--quan-mode',type=str,default="FP32",metavar='Q',
                        help='QUAN mode(default: FP32)')
    parser.add_argument('--batch-size',type=int,default=100,metavar='N',
                        help='batch size for training(default: 100)')
    parser.add_argument('--test-batch-size',type=int,default=100,metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs',type=int,default=150,metavar='N',
                        help='number of epoch to train(default: 150)')
    parser.add_argument('--lr-epochs',type=int,default=30,metavar='N',
                        help='number of epochs to decay learning rate(default: 30)')
    parser.add_argument('--lr',type=float,default=1e-1,metavar='LR',
                        help='learning rate(default: 1e-1)')
    parser.add_argument('--momentum',type=float,default=0.9,metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-5,metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed',type=int,default=1,metavar='S',
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval',type=int,default=50,metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-file-prefix',type=str,default="mnist",metavar='P',
                        help='')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def main():
    args = ParseArgs()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    logf = open(args.log_file_prefix+'_'+args.quan_mode+'.log', 'w')
    
    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    ###################################################################
    ##             Load Train Dataset                                ##
    ###################################################################
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./cifar100_data', train=True, download=True,
                   transform=transforms_train),
    batch_size=BATCH_SIZE, shuffle=True,**kwargs)
    ###################################################################
    ##             Load Test Dataset                                ##
    ###################################################################
    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./cifar100_data', train=False, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True,**kwargs)

    if args.quan_mode == "BINARY":
        model = QUANWNS.VGG7_BPWNs(100)
    elif args.quan_mode == "TERNARY":
        model = QUANWNS.VGG7_TWNs(100)
    else:
        model = QUANWNS.VGG7(100)

    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    
    optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    #optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    best_acc = 0.0 
    for epoch_index in range(1,args.epochs+1):
        lr=adjust_learning_rate(learning_rate,optimizer,epoch_index,args.lr_epochs)
        train(args,epoch_index,train_loader,model,optimizer,criterion,lr, logf)
        acc = test(args,model,test_loader,criterion, logf)
        if acc > best_acc:
            best_acc = acc
            U.save_model(model,best_acc,args.quan_mode)
            
    logf.write("best_acc: "+str(best_acc)+'\n')
    logf.close()

def train(args,epoch_index,train_loader,model,optimizer,criterion, lr=None, logf=None):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        if args.cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output,target)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            logss = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}@{:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), lr)
            print(logss)
            logf.write(logss + '\n')
            logf.flush()


def test(args,model,test_loader,criterion, logf=None):
    model.eval()
    test_loss = 0
    correct = 0

    for data,target in test_loader:
        if args.cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)
        output = model(data)
        test_loss += criterion(output,target).data.item()
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    acc = 100. * correct/len(test_loader.dataset)

    test_loss /= len(test_loader)
    logss = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    print(logss)
    logf.write(logss + '\n')
    logf.flush()
    
    return acc
    
def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr

if __name__ == '__main__':
    main()
