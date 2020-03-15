import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

# x = torch.linspace(-5, 5, 200)
# x = Variable(x)
# x_np = x.data.numpy()

# y_relu = torch.relu(x).data.numpy()
# y_sigmoid = torch.sigmoid(x).data.numpy()
# y_tanh = torch.tanh(x).data.numpy()
# # y_softmax = torch.softplus(x).data.numpy()

# plt.figure(1, figsize = (8, 6))
# plt.subplot(221)
# plt.plot(x_np, y_relu, c = 'red', label = 'relu')
# plt.ylim((-1, 5))
# plt.legend(loc = 'best')

# plt.subplot(222)
# plt.plot(x_np, y_sigmoid, c = 'red', label = 'sigmoid')
# plt.ylim((-0.2, 1.2))
# plt.legend(loc = 'best')

# plt.subplot(223)
# plt.plot(x_np, y_tanh, c = 'red', label = 'tanh')
# plt.ylim((-1.2, 1.2))
# plt.legend(loc = 'best')

# plt.show()

# ------------------------------------------------------------------------------

# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
# y = x.pow(2) + 0.2 * torch.rand(x.size())

# x,y = Variable(x), Variable(y)

# # plt.scatter(x.data.numpy(), y.data.numpy())
# # plt.show()

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, nhidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, nhidden)
#         self.preditc = torch.nn.Linear(nhidden, n_output)
    
#     def forward(self, x):
#         x = torch.relu(self.hidden(x))
#         x = self.preditc(x)
#         return x

# net = Net(1, 10, 1)
# print(net)

# optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
# lossfunc = torch.nn.MSELoss()

# for t in range(100):
#     prediction = net(x)
#     loss = lossfunc(prediction, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if t % 5 == 0 :
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy())
#         plt.text(0.5, 0, 'loss = %.4f' % loss.data.item(), fontdict = {'size': 20, 'color' :'red'})
#         plt.pause(0.1)
# plt.ioff()
# plt.show()  

# -------------------------------------------------------------------------------------------------------------
## 简单分类

# n_data = torch.ones(100,2)
# x0 = torch.normal(2 * n_data , 1)
# y0 = torch.zeros(100)
# # print(x0)

# x1 = torch.normal(-2 * n_data , 1)
# y1 = torch.ones(100)
# # print(x1)

# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
# y = torch.cat((y0, y1), ).type(torch.LongTensor)

# x, y = Variable(x), Variable(y)

# # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c = y.data.numpy(), s = 100, lw = 0)
# # plt.show()

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, nhidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, nhidden)
#         self.preditc = torch.nn.Linear(nhidden, n_output)
    
#     def forward(self, x):
#         x = torch.relu(self.hidden(x))
#         x = self.preditc(x)
#         return x

# net = Net(2, 10, 2)
# print(net)

# plt.ion()
# plt.show()
# optimizer = torch.optim.SGD(net.parameters(), lr = 0.02)
# lossfunc = torch.nn.CrossEntropyLoss()

# for t in range(100):
#     out = net(x)
#     # print(out.shape)
#     loss = lossfunc(out, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if t % 2 == 0:
#         plt.cla()
#         prediction = torch.max(torch.softmax(out, dim = 1), 1)[1]
#         pre_y = prediction.data.numpy().squeeze()
#         print(pre_y)
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c = pre_y, s = 100, lw = 0,cmap=plt.cm.get_cmap('RdYlBu'))
#         accu = sum(pre_y == target_y)/200
#         plt.text(1.5, -4, 'Accu = %.2f' % accu, fontdict = {'size': 20, 'color':'red'})
#         plt.pause(0.1)
# plt.ioff()
# plt.show() 


# -------------------------------------------------------------------------------------------------------------

##快速搭建框架

# n_data = torch.ones(100,2)
# x0 = torch.normal(2 * n_data , 1)
# y0 = torch.zeros(100)
# # print(x0)

# x1 = torch.normal(-2 * n_data , 1)
# y1 = torch.ones(100)
# # print(x1)

# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
# y = torch.cat((y0, y1), ).type(torch.LongTensor)

# x, y = Variable(x), Variable(y)

# # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c = y.data.numpy(), s = 100, lw = 0)
# # plt.show()

# net2 = torch.nn.Sequential(
#     torch.nn.Linear(2,10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10,2),
# )


# -------------------------------------------------------------------------------------------------------------

##保存网络 和提取网络
# torch.manual_seed(0)

# #fakec data
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
# y = x.pow(2) + 0.2 * torch.rand(x.size())
# x, y = Variable(x, requires_grad = False ),  Variable(y, requires_grad = False)

# def save():
#     net = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1),
#     )
#     optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
#     loss_func = torch.nn.MSELoss()
    
#     for t in range(100):
#         scores = net(x)
#         loss = loss_func(scores, y)
#         optimizer.zero_grad()
#         loss.backward() 
#         optimizer.step()

#     torch.save(net, 'net.pkl')#保存entire_net
#     torch.save(net.state_dict(), 'net_params.pkl')#保存params
#     y1 = net(x)
#     plt.figure(1, figsize = (10, 3))
#     plt.subplot(131)
#     plt.title('net1')
#     plt.scatter(x.data.numpy(), y.data.numpy())
#     plt.plot(x.data.numpy(), y1.data.numpy(), 'r-', lw = 5)

# def restore_net():
#     net2 = torch.load('net.pkl')
#     y2 = net2(x)
#     plt.subplot(132)
#     plt.title('net2')
#     plt.scatter(x.data.numpy(), y.data.numpy())
#     plt.plot(x.data.numpy(), y2.data.numpy(), 'r-', lw = 5)

# def restore_params():
#     net3 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1),        
#     )
#     net3.load_state_dict(torch.load('net_params.pkl'))
#     y3 = net3(x)
#     plt.subplot(133)
#     plt.title('net3')
#     plt.scatter(x.data.numpy(), y.data.numpy())
#     plt.plot(x.data.numpy(), y3.data.numpy(), 'r-', lw = 5)
#     plt.show()
# #保存网络
# save()
# #提取网络
# restore_net()
# # 提取参数
# restore_params()


# -------------------------------------------------------------------------------------------------------------
#批训练
# import torch.utils.data as Data

# BATCH_SIZE = 5

# x = torch.linspace(1, 10, 10)
# y = torch.linspace(10, 1, 10)

# torch_dataset = Data.TensorDataset(x, y)
# loader = Data.DataLoader(
#     dataset = torch_dataset,
#     batch_size = BATCH_SIZE,
#     shuffle = True,
#     num_workers = 0 ,
# )


# for epoch in range(3):   # 训练所有!整套!数据 3 次
#     for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
#         # 假设这里就是你训练的地方...
 
#         # 打出来一些数据
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#               batch_x.numpy(), '| batch y: ', batch_y.numpy())
# 

# -------------------------------------------------------------------------------------------------------------

# #优化器
# import torch.utils.data as Data

# #超参数 HYPER parameters
# LR = 0.01
# BATCH_SIZE = 32
# EPOCH = 12

# x = torch.unsqueeze(torch.linspace(-1, 1 , 1000), dim = 1)
# y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# torch_dataset = Data.TensorDataset(x,y)
# loader = Data.DataLoader(dataset = torch_dataset, batch_size = BATCH_SIZE, shuffle = True,)
# net_SGD = torch.nn.Sequential(
#     torch.nn.Linear(1, 20),
#     torch.nn.ReLU(),
#     torch.nn.Linear(20, 1),
# )
# net_Momentum = torch.nn.Sequential(
#     torch.nn.Linear(1, 20),
#     torch.nn.ReLU(),
#     torch.nn.Linear(20, 1),
# )
# net_RMSprop = torch.nn.Sequential(
#     torch.nn.Linear(1, 20),
#     torch.nn.ReLU(),
#     torch.nn.Linear(20, 1),
# )
# net_Adam = torch.nn.Sequential(
#     torch.nn.Linear(1, 20),
#     torch.nn.ReLU(),
#     torch.nn.Linear(20, 1),
# )
# nets=[net_SGD, net_Momentum, net_RMSprop, net_Adam]


# optimizer_SGD = torch.optim.SGD(net_SGD.parameters(), lr = LR)
# optimizer_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr = LR, momentum = 0.9)
# optimizer_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr = LR, alpha = 0.9)
# optimizer_Adam = torch.optim.Adam(net_Adam.parameters(), lr = LR, betas = (0.9, 0.99))
# optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_RMSprop, optimizer_Adam]
# lossfunc = torch.nn.MSELoss()
# LOSSES_HIS = [[], [], [], []]#记录损失
# for epoch in range(EPOCH):
#     print(epoch)
#     for step,(batch_x, batch_y) in enumerate(loader):
#         b_x = Variable(batch_x)
#         b_y = Variable(batch_y)
#         for net, opt , l_his in zip(nets, optimizers, LOSSES_HIS):
#             output = net(b_x)
#             loss = lossfunc(output, b_y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#             l_his.append(loss.data.item())
# labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
# for i, l_his in enumerate(LOSSES_HIS):
#     plt.plot(l_his, label = labels[i])
# plt.legend(loc = 'best')        
# plt.xlabel('step')
# plt.ylabel('Loss')
# plt.show()

# -------------------------------------------------------------------------------------------------------------
#CNN的尝试

# import torch.nn as nn
# from torch.autograd import Variable
# import torch.utils.data as Data
# import torchvision 
# import matplotlib.pyplot as plt

# # hyper parameters 
# EPOCH = 1 
# BANCH_SIZE = 50 
# LR = 0.001 
# download_mnist = False

# train_data = torchvision.datasets.MNIST(
#     root = './mnist',
#     train = True,
#     transform = torchvision.transforms.ToTensor(), #由于是黑白图片 所以每个像素的值只有一个，范围（0，1）
#     download = False
# )

# plot one example
# print(train_data.train_data.size())#(60000,28,28)
# print(train_data.train_labels.size())#(60000,1)
# plt.imshow(train_data.train_data[1].numpy(),cmap='gray')
# plt.title('%i' % train_data.train_labels[1])
# plt.show()

# train_loader = Data.DataLoader(dataset = train_data, batch_size = BANCH_SIZE, shuffle = True, num_workers = 0 )

# test_data = torchvision.datasets.MNIST(
#     root = './mnist',
#     train = False,
# )
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim = 1), volatile = True).type(torch.FloatTensor)[:2000]/255.
# test_y = test_data.targets[:2000]
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels = 1, 
#                 out_channels = 16,
#                 kernel_size = 5,
#                 stride = 1,
#                 padding= 2, #if stride = 1 , padding = (kernel-1)/2

#             ),
#             nn.ReLU(),#(16,28,28)
#             nn.MaxPool2d(
#                 kernel_size = 2,
#             ),#(16,14,14)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2),
#             nn.ReLU(),#(32,14,14)
#             nn.MaxPool2d(2),#(32,7,7)
#         )
#         self.af =nn.Linear(32 * 7 *7 , 10)
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x) #(batch, 32, 7 , 7)
#         x = x.view(x.size(0), -1) #(batch ,32 * 7 * 7)
#         output = self.af(x)
#         return output
# cnn = CNN()
# optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
# loss_func = nn.CrossEntropyLoss()


# for epoch in range(EPOCH): 
#     for step, (x, y) in enumerate(train_loader):
#         b_x = Variable(x)
#         b_y = Variable(y)
#         pred = cnn(b_x)
#         loss = loss_func(pred, b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if step % 50 == 0 :
#             test_output = cnn(test_x)
#             pred_y = torch.max(test_output, 1)[1].data.squeeze()
#             # print(test_y.size(0))
#             accurancy = torch.sum(pred_y == test_y).item()/test_y.size(0)
#             print('epoch',epoch,'|train loss %.4f' % loss.data.item(), '/test acc %.2f' % accurancy)
# test_output = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.squeeze()
# print(pred_y.numpy(),'prediction number')
# print(test_y[:10].numpy(),'real number')
            

# -------------------------------------------------------------------------------------------------------------
# #RNN尝试
# import torch
# from torch import nn
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import torch.utils.data as Data


# torch.manual_seed(1)    # reproducible

# # Hyper Parameters
# EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
# BATCH_SIZE = 50
# TIME_STEP = 28      # rnn 时间步数 / 图片高度
# INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素
# LR = 0.01           # learning rate
# DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle


# # Mnist 手写数字
# train_data = dsets.MNIST(
#     root='./mnist/',    # 保存或者提取位置
#     train=True,  # this is training data
#     transform=transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
#                                                     # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
#     download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
# )

# test_data = dsets.MNIST(root='./mnist/', train=False)

# # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# # print(train_data.shape)#无法输出shape mnist object
# # 为了节约时间, 我们测试时只测试前2000个
# test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# print(test_x.shape)
# test_y = test_data.targets[:2000]

# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN,self).__init__()

#         self.rnn = nn.LSTM(     # LSTM 效果要比 nn.RNN() 好多了
#             input_size=28,      # 图片每行的数据像素点
#             hidden_size=64,     # rnn hidden unit
#             num_layers=1,       # 有几层 RNN layers
#             batch_first=True,# input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            
#         )

#         self.out = nn.Linear(64, 10)    # 输出层
#     def forward(self, x):
#         # x shape (batch, time_step, input_size)
#         # r_out shape (batch, time_step, output_size)
#         # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
#         # h_c shape (n_layers, batch, hidden_size)
#         r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state

#         # 选取最后一个时间点的 r_out 输出
#         # 这里 r_out[:, -1, :] 的值也是 h_n 的值
#         out = self.out(r_out[:, -1, :])#r_out (batch,time step ,imput size)
#         return out

# rnn = RNN()
# # print(rnn)
# optimizer = torch.optim.Adam(rnn.parameters(), lr = LR)
# loss_func = nn.CrossEntropyLoss()
# for epoch in range(EPOCH):
#     for step ,(x, y) in enumerate(train_loader):
#         b_x = Variable(x.view(-1, 28, 28)) #reshape x to (batch, time_step ,input_size)
#         b_y = Variable(y)
#         output = rnn(b_x)
#         loss = loss_func(output, b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if step % 50 ==0 :
#             test_out = rnn(test_x.view(-1, 28, 28)) #(sample , time_step,imputsize)
#             pre_testy = torch.max(test_out, 1)[1].data.squeeze()
#             accuracy = torch.sum(pre_testy == test_y).numpy()/ test_y.size(0)
#             print('EPOCH:', epoch, '|train loss : %.4f' % loss.data, '|test accu= %.2f' % accuracy)
        
# test_output = rnn(test_x[:10].view(-1, 28, 28))
# pre_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pre_y)
# print(test_y[:10])




# -------------------------------------------------------------------------------------------------------------
#LSTM  做回归的任务

# import torch
# from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt

# # torch.manual_seed(1)    # reproducible

# # Hyper Parameters
# TIME_STEP = 10      # rnn time step
# INPUT_SIZE = 1      # rnn input size
# LR = 0.02           # learning rate

# # show data
# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()


# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()

#         self.rnn = nn.RNN(
#             input_size=INPUT_SIZE,
#             hidden_size=32,     # rnn hidden unit
#             num_layers=1,       # number of rnn layer
#             batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
#         )
#         self.out = nn.Linear(32, 1)

#     def forward(self, x, h_state):
#         # x (batch, time_step, input_size)
#         # h_state (n_layers, batch, hidden_size)
#         # r_out (batch, time_step, hidden_size)
#         r_out, h_state = self.rnn(x, h_state)
#         # print(r_out[:, 1, :].size())
#         outs = []    # save all predictions
#         for time_step in range(r_out.size(1)):    # calculate output for each time step
#             outs.append(self.out(r_out[:, time_step, :]))
#         return torch.stack(outs, dim=1), h_state

#         # instead, for simplicity, you can replace above codes by follows
#         # r_out = r_out.view(-1, 32)
#         # outs = self.out(r_out)
#         # outs = outs.view(-1, TIME_STEP, 1)
#         # return outs, h_state
        
#         # or even simpler, since nn.Linear can accept inputs of any dimension 
#         # and returns outputs with same dimension except for the last
#         # outs = self.out(r_out)
#         # return outs

# rnn = RNN()
# print(rnn)

# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
# loss_func = nn.MSELoss()

# h_state = None      # for initial hidden state

# plt.figure(1, figsize=(12, 5))
# plt.ion()           # continuously plot

# for step in range(100):
#     start, end = step * np.pi, (step+1)*np.pi   # time range
#     # use sin predicts cos
#     steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)  # float32 for converting torch FloatTensor
#     x_np = np.sin(steps)
#     y_np = np.cos(steps)

#     x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
#     y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

#     prediction, h_state = rnn(x, h_state)   # rnn output
#     # print(prediction.size())
#     # print(y_np.shape)
#     # !! next step is important !!
#     h_state = h_state.data        # repack the hidden state, break the connection from last iteration

#     loss = loss_func(prediction, y)         # calculate loss
#     optimizer.zero_grad()                   # clear gradients for this training step
#     loss.backward()                         # backpropagation, compute gradients
#     optimizer.step()                        # apply gradients
#     # print(np.size(y_np.flatten()))
#     # plotting
#     plt.plot(steps, y_np.flatten(), 'r-')
#     plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
#     plt.draw(); plt.pause(0.05)

# plt.ioff()
# plt.show()

# --------------------------------------------------------------------------------------------------------------------------
# #autoencode 自编码
# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# import torchvision
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import numpy as np

# #超参数
# EPOCH = 10
# BATCH_SIZE = 64
# LR = 0.005
# DOWNLOAD_MNIST = False
# N_TEST_IMG = 5

# train_data = torchvision.datasets.MNIST(
#     root= './mnist/',
#     train = True,
#     transform = torchvision.transforms.ToTensor(),
#     download = DOWNLOAD_MNIST,
# )
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder,self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 128),
#             nn.Tanh(),
#             nn.Linear(128, 64),
#             nn.Tanh(),
#             nn.Linear(64, 12),
#             nn.Tanh(),
#             nn.Linear(12, 3),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(3, 12),
#             nn.Tanh(),
#             nn.Linear(12, 64),
#             nn.Tanh(),
#             nn.Linear(64, 128),
#             nn.Tanh(),
#             nn.Linear(128, 28*28),
#             nn.Sigmoid(),
#         )
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded

# autoencoder = AutoEncoder()

# optimizer = torch.optim.Adam(autoencoder.parameters(), lr = LR)
# loss_func = nn.MSELoss()
# f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
# plt.ion()   # continuously plot

# # original data (first row) for viewing
# view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
# for i in range(N_TEST_IMG):
#     a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

# for epoch in range(EPOCH):
#     for step, (x, b_label) in enumerate(train_loader):
#         b_x = x.view(-1, 28*28)   # batch x, shape (batch, 28*28)
#         b_y = x.view(-1, 28*28)   # batch y, shape (batch, 28*28)

#         encoded, decoded = autoencoder(b_x)

#         loss = loss_func(decoded, b_y)      # mean square error
#         optimizer.zero_grad()               # clear gradients for this training step
#         loss.backward()                     # backpropagation, compute gradients
#         optimizer.step()                    # apply gradients

#         if step % 100 == 0:
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

#             # plotting decoded image (second row)
#             _, decoded_data = autoencoder(view_data)
#             for i in range(N_TEST_IMG):
#                 a[1][i].clear()
#                 a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
#                 a[1][i].set_xticks(()); a[1][i].set_yticks(())
#             plt.draw(); plt.pause(0.05)

# plt.ioff()
# plt.show()

# # visualize in 3D plot
# view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
# encoded_data, _ = autoencoder(view_data)
# fig = plt.figure(2); ax = Axes3D(fig)
# X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
# values = train_data.train_labels[:200].numpy()
# for x, y, z, s in zip(X, Y, Z, values):
#     c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor = c)
# ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
# plt.show()
# --------------------------------------------------------------------------------------------------------------------------

