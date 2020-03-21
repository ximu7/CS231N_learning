

import torch
import torch.nn as nn
from torch.autograd import Variable

x = torch.FloatTensor([0.1,1,10,100,1000,10000]).view(1,-1,1,1)
x = Variable(x)
print(x.data.size())
conv = nn.Conv2d(in_channels=6,
                 out_channels=6,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=3,
                 bias=False)

print(conv.weight.data.size())
# [1,2,3,4,5,6,7,8,9,10,11,12]
conv.weight.data =torch.arange(1,13).view(6,2,1,1).float()

# print(conv.weight.data)

output=conv(x)
print(output.data.size())