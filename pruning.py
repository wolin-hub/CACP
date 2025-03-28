import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path


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


def validate_model_structure(model):

    expected_layers = ['conv1', 'conv2']
    actual_layers = dict(model.named_modules()).keys()

    for layer in expected_layers:
        if layer not in actual_layers:
            raise ValueError(f"模型缺少关键层: {layer}")

    # 验证卷积层参数
    assert model.conv1.out_channels == 6, "conv1输出通道数应为6"
    assert model.conv2.out_channels == 16, "conv2输出通道数应为16"
    print("模型结构验证通过")


def prune_custom_model(model, composite_scores_file, prune_ratio=0.1, output_dir="unlearning_model"):

    # 验证模型结构
    validate_model_structure(model)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载得分数据
    df = pd.read_csv(composite_scores_file)
    target_class = df['Target_Class'].iloc[0]

    # 按层处理剪枝
    for layer_name in ['conv1', 'conv2']:
        # 过滤当前层数据
        layer_df = df[df['Layer'] == layer_name]

        if layer_df.empty:
            print(f"警告：{layer_name} 无得分数据，跳过")
            continue

        # 按综合得分排序（从高到低）
        sorted_df = layer_df.sort_values('Composite_Score', ascending=False)

        # 计算剪枝数量
        total_channels = sorted_df['Channel'].nunique()
        n_prune = max(1, int(total_channels * prune_ratio))  # 至少剪1个通道

        # 获取要剪枝的通道ID
        prune_channels = sorted_df.head(n_prune)['Channel'].tolist()

        # 获取卷积层
        conv_layer = getattr(model, layer_name)

        # 执行权重置零
        with torch.no_grad():
            # 处理权重 (out_channels, in_channels, H, W)
            conv_layer.weight[prune_channels] = 0.0

            # 处理偏置（如果存在）
            if conv_layer.bias is not None:
                conv_layer.bias[prune_channels] = 0.0

        print(f"已剪枝 {layer_name} 的以下通道: {prune_channels}")

    # 保存模型
    model_path = os.path.join(output_dir, f"pruned_net_{target_class}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"剪枝后模型已保存至: {model_path}")

    return model


if __name__ == "__main__":
    # 初始化模型
    model = Net()

    # 加载预训练权重（示例路径）
    model.load_state_dict(torch.load("./model/user_badepoch1_total_15rounds_cifar10.pt"))

    # 执行剪枝
    pruned_model = prune_custom_model(
        model=model,
        composite_scores_file="channel_composite_scores.csv",
        prune_ratio=0.1,  # 剪除各层前10%的高得分通道
        output_dir="unlearning_model"
    )