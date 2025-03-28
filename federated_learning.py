import os.path
from imaplib import Time2Internaldate
import torch
from time import process_time
# 用于构建NN
import torch.nn as nn
# 需要用到这个库里面的激活函数
import torch.nn.functional as F
# 用于构建优化器
import torch.optim as optim
# 用于初始化数据
from torchvision import datasets, transforms
# 用于分布式训练
import syft as sy

hook = sy.TorchHook(torch)
Bob = sy.VirtualWorker(hook, id='Bob')  # 远程打工人
Alice = sy.VirtualWorker(hook, id='Alice')
Cowboy = sy.VirtualWorker(hook, id='Cowboy')
d = sy.VirtualWorker(hook, id='d')
e = sy.VirtualWorker(hook, id='e')
f = sy.VirtualWorker(hook, id='f')
g = sy.VirtualWorker(hook, id='g')
h = sy.VirtualWorker(hook, id='h')
i = sy.VirtualWorker(hook, id='i')
j = sy.VirtualWorker(hook, id='j')



class Arguments:
    def __init__(self):
        self.dataset_name = 'cifar'
        self.batch_size = 128
        self.test_batch_size = 1000
        self.epochs = 15
        self.lr = 0.01
        self.momentum = 0.9
        # self.no_cuda = False
        self.seed = 1
        self.log_interval = 1
        self.save_model = True


# 实例化参数类
args = Arguments()

# 固定化随机数种子，使得每次训练的随机数都是固定的
torch.manual_seed(args.seed)
# cuda
# use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
# kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else ()
# 定义联邦训练数据集，定义转换器为 x=(x-mean)/标准差

data_root = os.path.abspath(os.path.join(os.getcwd(), "./train_dataset"))
image_path = os.path.join(data_root)
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


fed_dataset_Bob = datasets.ImageFolder(root=os.path.join(image_path, "train_2"),
                                       transform=transform)

# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_Bob = torch.utils.data.DataLoader(fed_dataset_Bob, batch_size=args.batch_size, shuffle=True)

# 定义联邦训练数据集，定义转换器为 x=(x-mean)/标准差
fed_dataset_Alice = datasets.ImageFolder(root=os.path.join(image_path, "train_1"),
                                         transform=transform)

# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_Alice = torch.utils.data.DataLoader(fed_dataset_Alice, batch_size=args.batch_size, shuffle=True)

fed_dataset_Cowboy = datasets.ImageFolder(root=os.path.join(image_path, "train_3"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_Cowboy = torch.utils.data.DataLoader(fed_dataset_Cowboy, batch_size=args.batch_size, shuffle=True)

fed_dataset_d = datasets.ImageFolder(root=os.path.join(image_path, "train_4"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_d = torch.utils.data.DataLoader(fed_dataset_d, batch_size=args.batch_size, shuffle=True)

fed_dataset_e = datasets.ImageFolder(root=os.path.join(image_path, "train_5"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_e = torch.utils.data.DataLoader(fed_dataset_e, batch_size=args.batch_size, shuffle=True)

fed_dataset_f = datasets.ImageFolder(root=os.path.join(image_path, "train_6"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_f = torch.utils.data.DataLoader(fed_dataset_f, batch_size=args.batch_size, shuffle=True)

fed_dataset_g = datasets.ImageFolder(root=os.path.join(image_path, "train_7"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_g = torch.utils.data.DataLoader(fed_dataset_g, batch_size=args.batch_size, shuffle=True)

fed_dataset_h = datasets.ImageFolder(root=os.path.join(image_path, "train_8"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_h = torch.utils.data.DataLoader(fed_dataset_h, batch_size=args.batch_size, shuffle=True)

fed_dataset_i = datasets.ImageFolder(root=os.path.join(image_path, "train_9"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_i = torch.utils.data.DataLoader(fed_dataset_i, batch_size=args.batch_size, shuffle=True)

fed_dataset_j = datasets.ImageFolder(root=os.path.join(image_path, "train_10"),
                                          transform=transform)
# 定义数据加载器，shuffle是采用随机的方式抽取数据
fed_loader_j = torch.utils.data.DataLoader(fed_dataset_j, batch_size=args.batch_size, shuffle=True)

# 定义测试集
test_dataset = datasets.ImageFolder(root=os.path.join(r"E:\mytest\mnist_test"), transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]))

# 定义测试集加载器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

n = len(fed_dataset_Bob) + len(fed_dataset_Alice) + len(fed_dataset_Cowboy) + len(fed_dataset_d) + len(fed_dataset_e)\
    + len(fed_dataset_f) + len(fed_dataset_g) + len(fed_dataset_h) + len(fed_dataset_i) + len(fed_dataset_j) # 数据集样本数量
n1 = len(fed_dataset_Alice)
n2 = len(fed_dataset_Bob)
n3 = len(fed_dataset_Cowboy)
n4 = len(fed_dataset_d)
n5 = len(fed_dataset_e)
n6 = len(fed_dataset_f)
n7 = len(fed_dataset_g)
n8 = len(fed_dataset_h)
n9 = len(fed_dataset_i)
n10 = len(fed_dataset_j)

# 构建神经网络模型
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc1 = nn.Linear(6 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        # x = x.view(-1, 6 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def fedavg_updata_weight(model: Net, Alice_model: Net, Bob_model: Net, Cowboy_model: Net, d_model:Net, e_model:Net, f_model:Net, g_model:Net,
                         h_model: Net, i_model: Net, j_model: Net,
                         ):
    """
    训练中需要修改的参数如下，对以下参数进行avg
    conv1.weight
    conv1.bias
    conv2.weight
    conv2.bias
    fc1.weight
    fc1.bias
    fc2.weight
    fc2.bias
    """
    if n == n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10:
        model.conv1.weight.set_(
            (
                    n1 * Alice_model.conv1.weight.data + n2 * Bob_model.conv1.weight.data + n3 * Cowboy_model.conv1.weight.data
            + n4 * d_model.conv1.weight.data + n5 * e_model.conv1.weight.data + n6 * f_model.conv1.weight.data + n7 * g_model.conv1.weight.data
            + n8 * h_model.conv1.weight.data + n9 * i_model.conv1.weight.data + n10 * j_model.conv1.weight.data) / n)
        # print('new:',model.conv1.weight.data)
        model.conv1.bias.set_(
            (
                    n1 * Alice_model.conv1.bias.data + n2 * Bob_model.conv1.bias.data + n3 * Cowboy_model.conv1.bias.data
            + n4 * d_model.conv1.bias.data + n5 * e_model.conv1.bias.data + n6 * f_model.conv1.bias.data + n7 * g_model.conv1.bias.data
            + n8 * h_model.conv1.bias.data + n9 * i_model.conv1.bias.data + n10 * j_model.conv1.bias.data) / n)
        model.conv2.weight.set_(
            (
                    n1 * Alice_model.conv2.weight.data + n2 * Bob_model.conv2.weight.data + n3 * Cowboy_model.conv2.weight.data
            + n4 * d_model.conv2.weight.data + n5 * e_model.conv2.weight.data + n6 * f_model.conv2.weight.data + n7 * g_model.conv2.weight.data
            + n8 * h_model.conv2.weight.data + n9 * i_model.conv2.weight.data + n10 * j_model.conv2.weight.data) / n)
        model.conv2.bias.set_(
            (
                    n1 * Alice_model.conv2.bias.data + n2 * Bob_model.conv2.bias.data + n3 * Cowboy_model.conv2.bias.data
            + n4 * d_model.conv2.bias.data + n5 * e_model.conv2.bias.data + n6 * f_model.conv2.bias.data + n7 * g_model.conv2.bias.data
            + n8 * h_model.conv2.bias.data + n9 * i_model.conv2.bias.data + n10 * j_model.conv2.bias.data) / n)
        model.fc1.weight.set_(
            (
                    n1 * Alice_model.fc1.weight.data + n2 * Bob_model.fc1.weight.data + n3 * Cowboy_model.fc1.weight.data
            + n4 * d_model.fc1.weight.data + n5 * e_model.fc1.weight.data + n6 * f_model.fc1.weight.data + n7 * g_model.fc1.weight.data
            + n8 * h_model.fc1.weight.data + n9 * i_model.fc1.weight.data + n10 * j_model.fc1.weight.data) / n)
        model.fc1.bias.set_(
            (
                    n1 * Alice_model.fc1.bias.data + n2 * Bob_model.fc1.bias.data + n3 * Cowboy_model.fc1.bias.data
            + n4 * d_model.fc1.bias.data + n5 * e_model.fc1.bias.data + n6 * f_model.fc1.bias.data + n7 * g_model.fc1.bias.data
            + n8 * h_model.fc1.bias.data + n9 * i_model.fc1.bias.data + n10 * j_model.fc1.bias.data) / n)
        model.fc2.weight.set_(
            (
                    n1 * Alice_model.fc2.weight.data + n2 * Bob_model.fc2.weight.data + n3 * Cowboy_model.fc2.weight.data
            + n4 * d_model.fc2.weight.data + n5 * e_model.fc2.weight.data + n6 * f_model.fc2.weight.data + n7 * g_model.fc2.weight.data
            + n8 * h_model.fc2.weight.data + n9 * i_model.fc2.weight.data + n10 * j_model.fc2.weight.data) / n)
        model.fc2.bias.set_(
            (
                    n1 * Alice_model.fc2.bias.data + n2 * Bob_model.fc2.bias.data + n3 * Cowboy_model.fc2.bias.data
            + n4 * d_model.fc2.bias.data + n5 * e_model.fc2.bias.data + n6 * f_model.fc2.bias.data + n7 * g_model.fc2.bias.data
            + n8 * h_model.fc2.bias.data + n9 * i_model.fc2.bias.data + n10 * j_model.fc2.bias.data) / n)

    else:
        print("False:n!=n1+n2+n3")


def train_abc():
    Bob_model = model.copy()
    Alice_model = model.copy()
    Cowboy_model = model.copy()
    d_model = model.copy()
    e_model = model.copy()
    f_model = model.copy()
    g_model = model.copy()
    h_model = model.copy()
    i_model = model.copy()
    j_model = model.copy()

    # 定义Bob的优化器
    Bob_opt = optim.SGD(Bob_model.parameters(), lr=args.lr, momentum=args.momentum)
    # 定义Alice的优化器
    Alice_opt = optim.SGD(Alice_model.parameters(), lr=args.lr, momentum=args.momentum)
    Cowboy_opt = optim.SGD(Cowboy_model.parameters(), lr=args.lr, momentum=args.momentum)
    d_opt = optim.SGD(d_model.parameters(), lr=args.lr, momentum=args.momentum)
    e_opt = optim.SGD(e_model.parameters(), lr=args.lr, momentum=args.momentum)
    f_opt = optim.SGD(f_model.parameters(), lr=args.lr, momentum=args.momentum)
    g_opt = optim.SGD(g_model.parameters(), lr=args.lr, momentum=args.momentum)
    h_opt = optim.SGD(h_model.parameters(), lr=args.lr, momentum=args.momentum)
    i_opt = optim.SGD(i_model.parameters(), lr=args.lr, momentum=args.momentum)
    j_opt = optim.SGD(j_model.parameters(), lr=args.lr, momentum=args.momentum)

    model.train()
    Bob_model.train()
    Alice_model.train()
    Cowboy_model.train()
    d_model.train()
    e_model.train()
    f_model.train()
    g_model.train()
    h_model.train()
    i_model.train()
    j_model.train()

    Bob_model.send(Bob)
    Alice_model.send(Alice)
    Cowboy_model.send(Cowboy)
    d_model.send(d)
    e_model.send(e)
    f_model.send(f)
    g_model.send(g)
    h_model.send(h)
    i_model.send(i)
    j_model.send(j)
    # 传递模型
    # Alice_loss = 8
    # Bob_loss = 8
    # 模拟Bob训练数据

    for epoch_ind, (data, target) in enumerate(fed_loader_Bob):
        data = data.send(Bob)
        target = target.send(Bob)
        # data, target = data.to(device), target.to(device)
        Bob_opt.zero_grad()
        pred = Bob_model(data)
        # print(pred.shape)
        # print(target.shape)
        Bob_loss = F.cross_entropy(pred, target)
        Bob_loss.backward()
        Bob_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in Bob loss:{:.6f}".format(epoch, epoch_ind,
                                                                             Bob_loss.get().data.item()))
            print("Bob batch:{} labels:{}".format(epoch_ind, target.get()))  # 打印标签

    # 模拟Alice训练模型
    for epoch_ind, (data, target) in enumerate(fed_loader_Alice):
        data = data.send(Alice)
        target = target.send(Alice)
        # data, target = data.to(device), target.to(device)

        Alice_opt.zero_grad()
        pred = Alice_model(data)
        Alice_loss = F.cross_entropy(pred, target)
        Alice_loss.backward()
        Alice_opt.step()
        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in Alice loss:{:.6f}".format(epoch, epoch_ind,
                                                                               Alice_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_Cowboy):
        data = data.send(Cowboy)
        target = target.send(Cowboy)
        # data, target = data.to(device), target.to(device)

        Cowboy_opt.zero_grad()
        pred = Cowboy_model(data)
        Cowboy_loss = F.cross_entropy(pred, target)
        Cowboy_loss.backward()
        Cowboy_opt.step()
        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in Cowboy loss:{:.6f}".format(epoch, epoch_ind,
                                                                                Cowboy_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_d):
        data = data.send(d)
        target = target.send(d)
        # data, target = data.to(device), target.to(device)
        d_opt.zero_grad()
        pred = d_model(data)
        # print(pred.shape)
        # print(target.shape)
        d_loss = F.cross_entropy(pred, target)
        d_loss.backward()
        d_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in d loss:{:.6f}".format(epoch, epoch_ind,
                                                                             d_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_e):
        data = data.send(e)
        target = target.send(e)
        # data, target = data.to(device), target.to(device)
        e_opt.zero_grad()
        pred = e_model(data)
        # print(pred.shape)
        # print(target.shape)
        e_loss = F.cross_entropy(pred, target)
        e_loss.backward()
        e_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in e loss:{:.6f}".format(epoch, epoch_ind,
                                                                             e_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_f):
        data = data.send(f)
        target = target.send(f)
        # data, target = data.to(device), target.to(device)
        f_opt.zero_grad()
        pred = f_model(data)
        # print(pred.shape)
        # print(target.shape)
        f_loss = F.cross_entropy(pred, target)
        f_loss.backward()
        f_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in d loss:{:.6f}".format(epoch, epoch_ind,
                                                                             f_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_g):
        data = data.send(g)
        target = target.send(g)
        # data, target = data.to(device), target.to(device)
        g_opt.zero_grad()
        pred = g_model(data)
        # print(pred.shape)
        # print(target.shape)
        g_loss = F.cross_entropy(pred, target)
        g_loss.backward()
        g_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in e loss:{:.6f}".format(epoch, epoch_ind,
                                                                             g_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_h):
        data = data.send(h)
        target = target.send(h)
        # data, target = data.to(device), target.to(device)
        h_opt.zero_grad()
        pred = h_model(data)
        # print(pred.shape)
        # print(target.shape)
        h_loss = F.cross_entropy(pred, target)
        h_loss.backward()
        h_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in d loss:{:.6f}".format(epoch, epoch_ind,
                                                                           h_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_i):
        data = data.send(i)
        target = target.send(i)
        # data, target = data.to(device), target.to(device)
        i_opt.zero_grad()
        pred = i_model(data)
        # print(pred.shape)
        # print(target.shape)
        i_loss = F.cross_entropy(pred, target)
        i_loss.backward()
        i_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in e loss:{:.6f}".format(epoch, epoch_ind,
                                                                           i_loss.get().data.item()))

    for epoch_ind, (data, target) in enumerate(fed_loader_j):
        data = data.send(j)
        target = target.send(j)
        # data, target = data.to(device), target.to(device)
        j_opt.zero_grad()
        pred = j_model(data)
        # print(pred.shape)
        # print(target.shape)
        j_loss = F.cross_entropy(pred, target)
        j_loss.backward()
        j_opt.step()  # 更新参数

        if epoch_ind % 50 == 0:
            print("There is epoch:{} epoch_ind:{} in d loss:{:.6f}".format(epoch, epoch_ind,
                                                                           j_loss.get().data.item()))



    with torch.no_grad():
        Bob_model.get()
        Alice_model.get()
        Cowboy_model.get()
        d_model.get()
        e_model.get()
        f_model.get()
        g_model.get()
        h_model.get()
        i_model.get()
        j_model.get()
        # 更新权重
        fedavg_updata_weight(model, Alice_model, Bob_model, Cowboy_model, d_model, e_model, f_model, g_model, h_model, i_model, j_model)
        # model.get()
    # print(Bob_model.conv1.weight.data)
    # print(Alice_model.conv1.weight.data)
    # print(model.conv1.weight.data)

    # 获得loss
    # 模型的loss
    # pred = model(fed_loader)
    # Loss = F.nll_loss(pred,target)
    # torch.save(Alice_model.state_dict(), "./model/10user_badeopch1_a_model_{}.pt".format(epoch))
    # torch.save(model.state_dict(), "./model/10user_badepoch1_total_{}.pt".format(epoch))
    print("Alice in train:")
    test(Alice_model, test_loader)
    print("Bob in train:")
    test(Bob_model, test_loader)
    print("Cowboy in train:")
    test(Cowboy_model, test_loader)
    print("d in train:")
    test(d_model, test_loader)
    print("e in train:")
    test(e_model, test_loader)
    print("f in train:")
    test(f_model, test_loader)
    print("g in train:")
    test(g_model, test_loader)
    print("h in train:")
    test(h_model, test_loader)
    print("i in train:")
    test(i_model, test_loader)
    print("j in train:")
    test(j_model, test_loader)
    # print("model in train:")
    # test(model, test_loader)


# 定义测试函数
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set : Average loss : {:.4f}, Accuracy: {}/{} ( {:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = Net()
    # print('pre:', model.conv1.weight.data)
    start = process_time()
    for epoch in range(1, args.epochs + 1):
        train_abc()
        print("{}model in test:".format(epoch))
        test(model, test_loader)
       # test_loader(model, test_loader)
        # model.get()
    torch.save(model.state_dict(), "./model/user_badepoch1_total_15rounds_cifar10.pt")
    print("model in test:")
    test(model, test_loader)
    end = process_time()

    time1 = end - start
    # time2 = end - mid
    # print("训练时间：{}h{}m{}s 测试时间为：{}h{}m{}s".format(time1 // 60 // 60, time1 // 60, time1 % 60, time2 // 60 // 60,
    #                                               time2 // 60, time2 % 60))

    print("训练时间：{}h{}m{}s ".format(time1 // 60 // 60, time1 // 60, time1 % 60, ))
