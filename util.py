import torch
import torch.nn as nn
import torch.nn.functional as F


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
        abssum = torch.matmul(absvalue, truth_value.to(torch.float32).view(-1,1))
        Alpha.append(abssum/count)
    
    alpha = torch.cat(Alpha, dim=0)
    return alpha

def Delta(tensor):
    n = tensor[0].nelement()
    if(len(tensor.size()) == 4):     #convolution layer
        delta = 0.75 * torch.sum(tensor.abs(), dim=(1,2,3))/n
    elif(len(tensor.size()) == 2):   #fc layer
        delta = 0.75 * torch.sum(tensor.abs(), dim=(1,))/n
    return delta

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
            
            
class Conv2DFunctionQUAN(torch.autograd.Function):
    def __init__(self):
        super(Conv2DFunctionQUAN, self).__init__()
        self.com_num = 0
        self.weight_fp32 = None
        
    @staticmethod
    def forward(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, quan_mode='TERANRY'):
        self.weight_fp32 = weight.data.clone().detach() #save a copy of fp32 precision weight
        if quan_mode == 'TERANRY':
            weight.data[:,:,:,:] = Ternarize(weight.data.clone().detach())[:,:,:,:] #do ternarization
        elif quan_mode == 'BINARY':
            weight.data[:,:,:,:] = Binarize(weight.data.clone().detach())[:,:,:,:] #do ternarization
        else:
            pass 
        
        self.save_for_backward(input, weight, bias)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        return output

    @staticmethod
    def backward(self, grad_output):
    
        input, weight, bias = self.saved_tensors
        stride, padding, dilation, groups = self.stride, self.padding, self.dilation, self.groups
        grad_input = grad_weight = grad_bias = grad_stride = grad_padding = grad_dilation = grad_groups = grad_quan_mode = None

        if self.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if self.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if self.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3), dtype=None, keepdim=False).squeeze(0) 
            
        self.saved_tensors[1].data[:,:,:,:] = self.weight_fp32[:,:,:,:] # recover the fp32 precision weight for parameter update
        
        return grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups, None

    
class TernaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(TernaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return Conv2DFunctionQUAN.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, 'TERANRY')
    
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return Conv2DFunctionQUAN.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, 'BINARY')


def save_model(model, acc, name_prefix='mnist'):
    print('Saving model ...')
    state = {
        'acc':acc,
        'state_dict':model.state_dict() 
    }
    torch.save(state, name_prefix+'-latest.pth')
    print('*** DONE! ***')
