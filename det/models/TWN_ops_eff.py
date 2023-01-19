from collections import OrderedDict
import math
import torch
from torch import nn
import torch.nn.functional as F


def Delta(tensor):
    n = tensor[0].nelement()
    if(len(tensor.size()) == 4):     #convolution layer
        delta = 0.70 * torch.sum(tensor.abs(), dim=(1,2,3))/n
    elif(len(tensor.size()) == 2):   #fc layer
        delta = 0.70 * torch.sum(tensor.abs(), dim=(1,))/n
    else:
        raise 'NO_IMPL'
    
    delta = torch.clamp(delta, 0, 100)
    
    return delta

def Alpha(tensor, delta):
    Alpha = []
    for i in range(tensor.size()[0]):
        count = 0
        abssum = 0
        absvalue = tensor[i].view(1,-1).abs()
        if isinstance(delta, int):
            truth_value = absvalue > delta
        else:
            truth_value = absvalue > delta[i]
            
        count = truth_value.sum()
        #print (count, truth_value.numel())
        #abssum = torch.matmul(absvalue, truth_value.to(torch.float32).view(-1,1))
        abssum = torch.matmul(absvalue, truth_value.to(absvalue.dtype).view(-1,1))
        Alpha.append(abssum/count)
    
    alpha = torch.cat(Alpha, dim=0)
    alpha = torch.clamp(alpha, 0, 100)
    
    return alpha


def Binarize(tensor):
    output = torch.zeros(tensor.size(), device=tensor.device)
    delta = 0
    alpha = Alpha(tensor, delta)
    for i in range(tensor.size()[0]):
        pos_one = (tensor[i] > delta).to(torch.float32)
        neg_one = pos_one-1
        out = torch.add(pos_one, neg_one)
        output[i] = torch.add(output[i], torch.mul(out, alpha[i]))
        
    return output

def Ternarize(tensor):
    output = torch.zeros(tensor.size(), device=tensor.device)
    delta = Delta(tensor)
    alpha = Alpha(tensor,delta)
    for i in range(tensor.size()[0]):
        pos_one = (tensor[i] > delta[i]).to(torch.float32)
        neg_one = -1 * (tensor[i] < -delta[i]).to(torch.float32)
        out = torch.add(pos_one, neg_one)
        output[i] = torch.add(output[i],torch.mul(out, alpha[i]))
        
    return output


def init_weight(*args):
    return nn.Parameter(nn.init.kaiming_normal_(torch.zeros(*args), mode='fan_out', nonlinearity='relu'))


class TernaryFunction(torch.autograd.Function):
    def __init__(self):
        super(TernaryFunction, self).__init__()
        
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x):
        return Ternarize(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):
        return g
    
class TernaryModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_weight(self, name):
        w = getattr(self, name)
        return TernaryFunction.apply(w)

    def forward(self):
        pass
    
    
class TernaryConv2d(TernaryModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(TernaryConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.register_parameter('conv_ternary', init_weight(out_channels, in_channels, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_channels))
            self.register_parameter('bias', self.bias)
        else:
            self.bias=None

    def forward(self, x):
        return F.conv2d(x, self._get_weight('conv_ternary'), bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
       

class BinaryFunction(torch.autograd.Function):
    def __init__(self):
        super(BinaryFunction, self).__init__()
        
    @staticmethod
    def forward(ctx, x):
        return Binarize(x)

    @staticmethod
    def backward(ctx, g):
        return g
    
class BinaryModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _get_weight(self, name):
        w = getattr(self, name)
        return BinaryFunction.apply(w)

    def forward(self):
        pass
    
    
class BinaryConv2d(BinaryModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BinaryConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.register_parameter('conv_binary', init_weight(out_channels, in_channels, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_channels))
            self.register_parameter('bias', self.bias)
        else:
            self.bias=None

    def forward(self, x):
        return F.conv2d(x, self._get_weight('conv_binary'), bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
      
      
def save_model(model, acc, name_prefix='quan'):
    print('Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state, name_prefix+'-latest.pth')
    print('*** DONE! ***')
