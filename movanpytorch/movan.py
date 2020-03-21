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
# 简单分类

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

import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# hyper parameters
EPOCH = 1
BANCH_SIZE = 50
LR = 0.001
download_mnist = False

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),  #由于是黑白图片 所以每个像素的值只有一个，范围（0，1）
    download=False)

print(train_data.train_data.size())  #(60000,28,28)
print(train_data.train_labels.size())  #(60000,1)
plt.imshow(train_data.train_data[1].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[1])
plt.show()

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=BANCH_SIZE,
                               shuffle=True,
                               num_workers=0)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
    torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,  #if stride = 1 , padding = (kernel-1)/2
            ),
            nn.ReLU(),  #(16,28,28)
            nn.MaxPool2d(kernel_size=2, ),  #(16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),  #(32,14,14)
            nn.MaxPool2d(2),  #(32,7,7)
        )
        self.af = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  #(batch, 32, 7 , 7)
        x = x.view(x.size(0), -1)  #(batch ,32 * 7 * 7)
        output = self.af(x)
        return output


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x
        b_y = y
        pred = cnn(b_x)
        loss = loss_func(pred, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            # print(test_y.size(0))
            accurancy = torch.sum(pred_y == test_y).item() / test_y.size(0)
            print('epoch', epoch, '|train loss %.4f' % loss.data.item(),
                  '/test acc %.2f' % accurancy)
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y.numpy(), 'prediction number')
print(test_y[:10].numpy(), 'real number')

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
# #DQN学习
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import gym

# # Hyper Parameters
# BATCH_SIZE = 32
# LR = 0.01                   # learning rate
# EPSILON = 0.9               # greedy policy
# GAMMA = 0.9                 # reward discount
# TARGET_REPLACE_ITER = 100   # target update frequency
# MEMORY_CAPACITY = 2000
# env = gym.make('CartPole-v0')
# env = env.unwrapped
# N_ACTIONS = env.action_space.n
# N_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# class Net(nn.Module):
#     def __init__(self, ):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(N_STATES, 50)
#         self.fc1.weight.data.normal_(0, 0.1)   # initialization
#         self.out = nn.Linear(50, N_ACTIONS)
#         self.out.weight.data.normal_(0, 0.1)   # initialization

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value

# class DQN(object):
#     def __init__(self):
#         self.eval_net, self.target_net = Net(), Net()

#         self.learn_step_counter = 0                                     # for target updating
#         self.memory_counter = 0                                         # for storing memory
#         self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
#         self.loss_func = nn.MSELoss()

#     def choose_action(self, x):
#         x = torch.unsqueeze(torch.FloatTensor(x), 0)
#         # input only one sample
#         if np.random.uniform() < EPSILON:   # greedy
#             actions_value = self.eval_net.forward(x)
#             action = torch.max(actions_value, 1)[1].data.numpy()
#             action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
#         else:   # random
#             action = np.random.randint(0, N_ACTIONS)
#             action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
#         return action

#     def store_transition(self, s, a, r, s_):
#         transition = np.hstack((s, [a, r], s_))
#         # replace the old memory with new memory
#         index = self.memory_counter % MEMORY_CAPACITY
#         self.memory[index, :] = transition
#         self.memory_counter += 1

#     def learn(self):
#         # target parameter update
#         if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
#         self.learn_step_counter += 1

#         # sample batch transitions
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
#         b_memory = self.memory[sample_index, :]
#         b_s = torch.FloatTensor(b_memory[:, :N_STATES])
#         b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
#         b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
#         b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

#         # q_eval w.r.t the action in experience
#         q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
#         q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
#         q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
#         loss = self.loss_func(q_eval, q_target)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

# dqn = DQN()

# print('\nCollecting experience...')
# for i_episode in range(400):
#     s = env.reset()
#     ep_r = 0
#     while True:
#         env.render()
#         a = dqn.choose_action(s)

#         # take action
#         s_, r, done, info = env.step(a)

#         # modify the reward
#         x, x_dot, theta, theta_dot = s_
#         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
#         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#         r = r1 + r2

#         dqn.store_transition(s, a, r, s_)

#         ep_r += r
#         if dqn.memory_counter > MEMORY_CAPACITY:
#             dqn.learn()
#             if done:
#                 print('Ep: ', i_episode,
#                       '| Ep_r: ', round(ep_r, 2))

#         if done:
#             break
#         s = s_

# --------------------------------------------------------------------------------------------------------------------------
# # #GAN学习
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# # torch.manual_seed(1)    # reproducible
# # np.random.seed(1)

# # Hyper Parameters
# BATCH_SIZE = 64
# LR_G = 0.0001           # learning rate for generator
# LR_D = 0.0001           # learning rate for discriminator
# N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
# ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
# PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# # show our beautiful painting range
# # plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# # plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# # plt.legend(loc='upper right')
# # plt.show()

# def artist_works():     # painting from the famous artist (real target)
#     a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
#     paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
#     paintings = torch.from_numpy(paintings).float()
#     return paintings

# G = nn.Sequential(                      # Generator
#     nn.Linear(N_IDEAS, 128),            # random ideas (could from normal distribution)
#     nn.ReLU(),
#     nn.Linear(128, ART_COMPONENTS),     # making a painting from these random ideas
# )

# D = nn.Sequential(                      # Discriminator
#     nn.Linear(ART_COMPONENTS, 128),     # receive art work either from the famous artist or a newbie like G
#     nn.ReLU(),
#     nn.Linear(128, 1),
#     nn.Sigmoid(),                       # tell the probability that the art work is made by artist
# )

# opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
# opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

# plt.ion()   # something about continuous plotting

# for step in range(10000):
#     artist_paintings = artist_works()           # real painting from artist
#     G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)  # random ideas
#     G_paintings = G(G_ideas)                    # fake painting from G (random ideas)

#     prob_artist0 = D(artist_paintings)          # D try to increase this prob
#     prob_artist1 = D(G_paintings)               # D try to reduce this prob

#     D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
#     G_loss = torch.mean(torch.log(1. - prob_artist1))

#     opt_D.zero_grad()
#     D_loss.backward(retain_graph=True)      # reusing computational graph
#     opt_D.step()

#     opt_G.zero_grad()
#     G_loss.backward()
#     opt_G.step()

#     if step % 50 == 0:  # plotting
#         plt.cla()
#         plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
#         plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
#         plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
#         plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 13})
#         plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
#         plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=10);plt.draw();plt.pause(0.01)

# plt.ioff()
# plt.show()
# --------------------------------------------------------------------------------------------------------------------------
# # # GPU CUDA
# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# import torchvision

# # torch.manual_seed(1)

# EPOCH = 1
# BATCH_SIZE = 50
# LR = 0.001
# DOWNLOAD_MNIST = False

# train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST,)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# # !!!!!!!! Change in here !!!!!!!!! #
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
# test_y = test_data.test_labels[:2000].cuda()

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
#                                    nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
#         self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
#         self.out = nn.Linear(32 * 7 * 7, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         output = self.out(x)
#         return output

# cnn = CNN()

# # !!!!!!!! Change in here !!!!!!!!! #
# cnn.cuda()      # Moves all model parameters and buffers to the GPU.

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()

# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):

#         # !!!!!!!! Change in here !!!!!!!!! #
#         b_x = x.cuda()    # Tensor on GPU
#         b_y = y.cuda()    # Tensor on GPU

#         output = cnn(b_x)
#         loss = loss_func(output, b_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if step % 50 == 0:
#             test_output = cnn(test_x)

#             # !!!!!!!! Change in here !!!!!!!!! #
#             pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

#             accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

# test_output = cnn(test_x[:10])

# # !!!!!!!! Change in here !!!!!!!!! #
# pred_y = torch.max(test_output, 1)[1].cuda().data # move the computation in GPU

# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')

# --------------------------------------------------------------------------------------------------------------------------
# # # DROPOUT 比较

# import torch
# import matplotlib.pyplot as plt

# # torch.manual_seed(1)    # reproducible

# N_SAMPLES = 20
# N_HIDDEN = 300

# # training data
# x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
# y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# # test data
# test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
# test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# # show data
# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

# net_overfitting = torch.nn.Sequential(
#     torch.nn.Linear(1, N_HIDDEN),
#     torch.nn.ReLU(),
#     torch.nn.Linear(N_HIDDEN, N_HIDDEN),
#     torch.nn.ReLU(),
#     torch.nn.Linear(N_HIDDEN, 1),
# )

# net_dropped = torch.nn.Sequential(
#     torch.nn.Linear(1, N_HIDDEN),
#     torch.nn.Dropout(0.5),  # drop 50% of the neuron
#     torch.nn.ReLU(),
#     torch.nn.Linear(N_HIDDEN, N_HIDDEN),
#     torch.nn.Dropout(0.5),  # drop 50% of the neuron
#     torch.nn.ReLU(),
#     torch.nn.Linear(N_HIDDEN, 1),
# )

# print(net_overfitting)  # net architecture
# print(net_dropped)

# optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
# optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
# loss_func = torch.nn.MSELoss()

# plt.ion()   # something about plotting

# for t in range(500):
#     pred_ofit = net_overfitting(x)
#     pred_drop = net_dropped(x)
#     loss_ofit = loss_func(pred_ofit, y)
#     loss_drop = loss_func(pred_drop, y)

#     optimizer_ofit.zero_grad()
#     optimizer_drop.zero_grad()
#     loss_ofit.backward()
#     loss_drop.backward()
#     optimizer_ofit.step()
#     optimizer_drop.step()

#     if t % 10 == 0:
#         # change to eval mode in order to fix drop out effect
#         net_overfitting.eval()##！！！！！！！！！！！！！！！！！！！！
#         net_dropped.eval()  # parameters for dropout differ from train mode

#         # plotting
#         plt.cla()
#         test_pred_ofit = net_overfitting(test_x)
#         test_pred_drop = net_dropped(test_x)
#         plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
#         plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
#         plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
#         plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
#         plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 20, 'color':  'red'})
#         plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
#         plt.legend(loc='upper left'); plt.ylim((-2.5, 2.5));plt.pause(0.1)

#         # change back to train mode
#         net_overfitting.train()##！！！！！！！！！！！！！！！！！！！！
#         net_dropped.train()

# plt.ioff()
# plt.show()

# --------------------------------------------------------------------------------------------------------------------------
# # # DROPOUT 比较
# import torch
# from torch import nn
# from torch.nn import init
# import torch.utils.data as Data
# import matplotlib.pyplot as plt
# import numpy as np

# # torch.manual_seed(1)    # reproducible
# # np.random.seed(1)

# # Hyper parameters
# N_SAMPLES = 2000
# BATCH_SIZE = 64
# EPOCH = 12
# LR = 0.03
# N_HIDDEN = 8
# ACTIVATION = torch.tanh
# B_INIT = -0.2   # use a bad bias constant initializer

# # training data
# x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
# noise = np.random.normal(0, 2, x.shape)
# y = np.square(x) - 5 + noise

# # test data
# test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
# noise = np.random.normal(0, 2, test_x.shape)
# test_y = np.square(test_x) - 5 + noise

# train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
# test_x = torch.from_numpy(test_x).float()
# test_y = torch.from_numpy(test_y).float()

# train_dataset = Data.TensorDataset(train_x, train_y)
# train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,)

# # show data
# plt.scatter(train_x.numpy(), train_y.numpy(), c='#FF9359', s=50, alpha=0.2, label='train')
# plt.legend(loc='upper left')

# class Net(nn.Module):
#     def __init__(self, batch_normalization=False):
#         super(Net, self).__init__()
#         self.do_bn = batch_normalization
#         self.fcs = []
#         self.bns = []
#         self.bn_input = nn.BatchNorm1d(1, momentum=0.5)   # for input data

#         for i in range(N_HIDDEN):               # build hidden layers and BN layers
#             input_size = 1 if i == 0 else 10
#             fc = nn.Linear(input_size, 10)
#             setattr(self, 'fc%i' % i, fc)       # IMPORTANT set layer to the Module
#             self._set_init(fc)                  # parameters initialization
#             self.fcs.append(fc)
#             if self.do_bn:
#                 bn = nn.BatchNorm1d(10, momentum=0.5)
#                 setattr(self, 'bn%i' % i, bn)   # IMPORTANT set layer to the Module
#                 self.bns.append(bn)

#         self.predict = nn.Linear(10, 1)         # output layer
#         self._set_init(self.predict)            # parameters initialization

#     def _set_init(self, layer):
#         init.normal_(layer.weight, mean=0., std=.1)
#         init.constant_(layer.bias, B_INIT)

#     def forward(self, x):
#         pre_activation = [x]
#         if self.do_bn: x = self.bn_input(x)     # input batch normalization
#         layer_input = [x]
#         for i in range(N_HIDDEN):
#             x = self.fcs[i](x)
#             pre_activation.append(x)
#             if self.do_bn: x = self.bns[i](x)   # batch normalization
#             x = ACTIVATION(x)
#             layer_input.append(x)
#         out = self.predict(x)
#         return out, layer_input, pre_activation

# nets = [Net(batch_normalization=False), Net(batch_normalization=True)]

# # print(*nets)    # print net architecture

# opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]

# loss_func = torch.nn.MSELoss()

# def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
#     for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
#         [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
#         if i == 0:
#             p_range = (-7, 10);the_range = (-7, 10)
#         else:
#             p_range = (-4, 4);the_range = (-1, 1)
#         ax_pa.set_title('L' + str(i))
#         ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5);ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
#         ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359');ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
#         for a in [ax_pa, ax, ax_pa_bn, ax_bn]: a.set_yticks(());a.set_xticks(())
#         ax_pa_bn.set_xticks(p_range);ax_bn.set_xticks(the_range)
#         axs[0, 0].set_ylabel('PreAct');axs[1, 0].set_ylabel('BN PreAct');axs[2, 0].set_ylabel('Act');axs[3, 0].set_ylabel('BN Act')
#     plt.pause(0.01)

# if __name__ == "__main__":
#     f, axs = plt.subplots(4, N_HIDDEN + 1, figsize=(10, 5))
#     plt.ion()  # something about plotting
#     plt.show()

#     # training
#     losses = [[], []]  # recode loss for two networks

#     for epoch in range(EPOCH):
#         print('Epoch: ', epoch)
#         layer_inputs, pre_acts = [], []
#         for net, l in zip(nets, losses):
#             net.eval()              # set eval mode to fix moving_mean and moving_var
#             pred, layer_input, pre_act = net(test_x)
#             l.append(loss_func(pred, test_y).data.item())
#             layer_inputs.append(layer_input)
#             pre_acts.append(pre_act)
#             net.train()             # free moving_mean and moving_var
#         plot_histogram(*layer_inputs, *pre_acts)     # plot histogram

#         for step, (b_x, b_y) in enumerate(train_loader):
#             for net, opt in zip(nets, opts):     # train for each network
#                 pred, _, _ = net(b_x)
#                 loss = loss_func(pred, b_y)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()    # it will also learns the parameters in Batch Normalization

#     plt.ioff()

#     # plot training loss
#     plt.figure(3)
#     plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
#     plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
#     plt.xlabel('step');plt.ylabel('test loss');plt.ylim((0, 2000));plt.legend(loc='best')
#     plt.show()
#     # evaluation
#     # set net to eval mode to freeze the parameters in batch normalization layers
#     [net.eval() for net in nets]    # set eval mode to fix moving_mean and moving_var
#     preds = [net(test_x)[0] for net in nets]
#     plt.figure(4)
#     plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
#     plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
#     plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
#     plt.legend(loc='best')
#     plt.show()