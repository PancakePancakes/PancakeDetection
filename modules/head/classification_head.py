from torch import nn
from torch.nn import functional as F


class FcClassificationHead(nn.Module):
    """全连接分类头网络。

    这个类实现了一个用于分类任务的全连接网络头，包含全局平均池化层和全连接层。

    网络结构:
        - 全局平均池化层: 将输入的特征图缩小到每个通道的单个值。
        - 全连接层: 将池化后的特征映射到指定的类别数。

    参数量:
        - 全连接层参数量: 输入通道数 x 类别数 + 类别数 (偏置项)

    输入形状:
        - (B, C, H, W): B 是批量大小，C 是通道数，H 和 W 是特征图的高度和宽度。

    输出形状:
        - (B, num_classes): B 是批量大小，num_classes 是类别数。

    Args:
        input_channels (int): 输入特征图的通道数。
        num_classes (int): 分类任务的类别数。

    """

    def __init__(self, input_channels, num_classes):
        """初始化函数。

        Args:
            input_channels (int): 输入特征图的通道数。
            num_classes (int): 分类任务的类别数。
        """
        super(FcClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出张量，形状为 (B, num_classes)。
        """
        # Assuming x is of shape (B, C, H, W)
        x = self.global_avg_pool(x)  # Pool to shape (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to shape (B, C)
        x = self.fc(x)  # Linear layer to shape (B, num_classes)
        return x


class SingleClassificationHead(FcClassificationHead):
    """单分类头网络。

    这个类继承自 FcClassificationHead，用于单标签分类任务，包含一个 Softmax 激活函数。

    输出形状:
        - (B, num_classes): B 是批量大小，num_classes 是类别数。

    """

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出张量，形状为 (B, num_classes)，经过 Softmax 激活。
        """
        x = super().forward(x)
        return F.softmax(x, dim=1)


class MultiClassificationHead(FcClassificationHead):
    """多分类头网络。

    这个类继承自 FcClassificationHead，用于多标签分类任务，包含一个 Sigmoid 激活函数。

    输出形状:
        - (B, num_classes): B 是批量大小，num_classes 是类别数。

    """

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            torch.Tensor: 输出张量，形状为 (B, num_classes)，经过 Sigmoid 激活。
        """
        x = super().forward(x)
        return F.sigmoid(x)
