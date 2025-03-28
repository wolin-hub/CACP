import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
from tqdm import tqdm


class FeatureExtractor:

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []

        # 注册前向钩子
        for layer_name in self.target_layers:
            layer = self._get_layer(layer_name)
            hook = layer.register_forward_hook(self._save_features(layer_name))
            self.hooks.append(hook)

    def _get_layer(self, layer_name):#递归获取指定层

        modules = dict([*self.model.named_modules()])
        return modules[layer_name]

    def _save_features(self, layer_name):#保存特征图的钩子函数


        def hook(module, input, output):
            self.features[layer_name] = output.detach()

        return hook

    def remove_hooks(self):#移除所有钩子

        for hook in self.hooks:
            hook.remove()


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def calculate_channel_similarity(model, data_dir, target_layers, output_file="similarity.csv"):

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 数据预处理（根据模型输入尺寸调整）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 根据模型实际训练尺寸调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 根据实际训练参数调整
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # 初始化特征提取器
    extractor = FeatureExtractor(model, target_layers)

    # 第一阶段：计算每个类别的平均特征图
    print("Calculating class average features...")
    class_features = {name: {layer: [] for layer in target_layers} for name in class_names}

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            _ = model(images)

            # 处理每个样本
            for i in range(len(labels)):
                label = class_names[labels[i]]
                for layer in target_layers:
                    feat = extractor.features[layer][i].cpu()  # (C, H, W)
                    class_features[label][layer].append(feat)

    # 计算平均特征图
    class_avg = {name: {} for name in class_names}
    for name in class_names:
        for layer in target_layers:
            class_avg[name][layer] = torch.stack(class_features[name][layer]).mean(dim=0)

    # 第二阶段：计算通道相似度
    print("Calculating channel similarities...")
    similarity_results = []

    # 重新遍历数据集计算相似度
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            _ = model(images)

            for i in range(len(labels)):
                label = class_names[labels[i]]
                for layer in target_layers:
                    # 当前样本的特征图
                    feat = extractor.features[layer][i].cpu()  # (C, H, W)
                    # 类别平均特征图
                    avg_feat = class_avg[label][layer]

                    # 计算每个通道的相似度
                    C = feat.size(0)
                    for ch in range(C):
                        # 展平为向量
                        vec_sample = feat[ch].flatten()
                        vec_avg = avg_feat[ch].flatten()

                        # 计算余弦相似度
                        similarity = F.cosine_similarity(vec_sample.unsqueeze(0), vec_avg.unsqueeze(0), dim=1).item()

                        similarity_results.append({
                            "Layer": layer,
                            "Class": label,
                            "Channel": ch,
                            "Similarity": similarity
                        })

    # 计算平均相似度
    df = pd.DataFrame(similarity_results)
    result = df.groupby(['Layer', 'Class', 'Channel']).mean().reset_index()

    # 保存结果
    result.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    extractor.remove_hooks()


if __name__ == "__main__":
    # 初始化并加载训练好的模型
    model = Net()
    model.load_state_dict(torch.load("./model/user_badepoch1_total_15rounds_cifar10.pt"))  # 加载训练好的权重

    # 设置要分析的卷积层
    target_layers = ['conv1', 'conv2']  # 对应网络中的两个卷积层

    # 运行分析
    calculate_channel_similarity(
        model=model,
        data_dir="./mnist_test",
        target_layers=target_layers,
        output_file="channel_similarities.csv"
    )