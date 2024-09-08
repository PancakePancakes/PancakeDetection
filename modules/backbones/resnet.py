from typing import Optional, Callable, Type, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicResidualBlock(nn.Module):
    """
    实现 ResNet 中的基本残差块。

    参考:
        https://arxiv.org/abs/1512.03385 (en) (Original paper)

    结构:
        - 3x3 卷积层：保持输入输出通道数一致。
        - 3x3 卷积层：进一步处理特征。
        - 可选的 1x1 卷积层：当输入输出通道数或步幅不匹配时，用于调整残差路径。

    批注:
        基本残差块通过两个 3x3 卷积层实现特征提取。
        如果 `stride` 不为 1 或 `input_channels` 不等于 `output_channels`，则会使用 1x1 卷积调整残差。
    """
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            stride: int = 1,
            activation_function: Optional[Callable] = None
    ) -> None:
        """
        初始化基本残差块。

        Args:
            input_channels (int): 输入通道数。
            output_channels (int): 输出通道数。
            stride (int, optional): 第一层卷积的步长，默认为 1。
            activation_function (Callable, optional): 激活函数，默认为 ReLU。
        """
        super(BasicResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)

        self.act_fn = activation_function or F.relu

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        # 当输入通道数不等于输出通道数时，需要添加卷积层
        # 当步幅不等于 1 时，需要进行下采样
        self.conv1x1 = None
        if stride != 1 or input_channels != output_channels:
            self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 生成快速连接
        if self.conv1x1 is not None:
            residual = self.conv1x1(x)
        else:
            residual = x

        # 两组卷积层
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 合并残差
        out += residual
        out = self.act_fn(out)
        return out


class BottleneckResidualBlock(nn.Module):
    """
    实现 ResNet 中的瓶颈残差块。

    参考:
        https://arxiv.org/abs/1512.03385 (en) (Original paper)

    结构:
        - 1x1 卷积层：将输入通道数减少为 `process_channels`。
        - 3x3 卷积层：在 `process_channels` 上进行卷积。
        - 1x1 卷积层：将通道数增加到 `output_channels`。
        - 可选的 1x1 卷积层：当输入输出通道数或步幅不匹配时，用于调整残差路径。

    批注:
        瓶颈块通过减少中间卷积层的通道数，减少了参数量和计算量。
        如果 `stride` 不为 1 或 `input_channels` 不等于 `output_channels`，则会使用 1x1 卷积调整残差。
    """
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            process_channels: int = None,
            stride: int = 1,
            activation_function: Optional[Callable] = None
    ) -> None:
        """
        初始化瓶颈残差块。

        Args:
            input_channels (int): 输入通道数。
            output_channels (int): 输出通道数。
            process_channels (int, optional): 中间处理通道数，默认为输入通道数的四分之一。
            stride (int, optional): 3x3 卷积的步长，默认为 1。
            activation_function (Callable, optional): 激活函数，默认为 ReLU。
        """
        super(BottleneckResidualBlock, self).__init__()

        process_channels = process_channels or input_channels // 4

        self.act_fn = activation_function or F.relu

        self.conv1 = nn.Conv2d(input_channels, process_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(process_channels)

        self.conv2 = nn.Conv2d(process_channels, process_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(process_channels)

        self.conv3 = nn.Conv2d(process_channels, output_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels)

        # 当输入通道数不等于输出通道数，或步幅不等于1时，需要添加卷积层
        self.conv1x1 = None
        if stride != 1 or input_channels != output_channels:
            self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        # 生成快速连接
        if self.conv1x1 is not None:
            residual = self.conv1x1(x)
        else:
            residual = x

        # 三组卷积层
        out = self.act_fn(self.bn1(self.conv1(x)))
        out = self.act_fn(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # 合并残差
        out += residual
        out = F.relu(out)
        return out


# 创建通用别名
ResidualBlock = BottleneckResidualBlock


class ResNet(nn.Module):
    """
    ResNet: Residual Networks

    参考:
    https://arxiv.org/abs/1512.03385 (en)(Original paper)
    https://zh.d2l.ai/chapter_convolutional-modern/resnet.html (zh-CN)

    结构：
    - Conv1: 7x7卷积, 64 filters, stride 2
    - Max Pooling: 3x3 pooling, stride 2
    - 多个残差层
    - Global Average Pooling

    批注: 使用BasicResidualBlock或BottleneckResidualBlock构建网络层
    """

    def __init__(
            self,
            num_blocks: list,
            input_channels: int = 3,
            conv1_channels: int = 64,
            block: Type[Union[BasicResidualBlock, BottleneckResidualBlock]] = BottleneckResidualBlock,
            output_channels_list: Optional[list] = (64, 128, 256, 512),
    ) -> None:
        """
        初始化ResNet模型。

        Args:
            num_blocks (list): 每个层级中残差块的数量。
            conv1_channels (int, optional): 第一个卷积层的输出通道数，默认64。
            input_channels (int, optional): 输入通道数，默认3。
            block (Type[Union[BasicResidualBlock, BottleneckResidualBlock]]): 残差块的类型。
            output_channels_list (list, optional): 每个层级的输出通道数，默认(64, 128, 256, 512)。
        """
        super(ResNet, self).__init__()
        self.in_channels = conv1_channels

        self.conv1 = nn.Conv2d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用nn.ModuleList来存储各层
        self.layers = nn.ModuleList()
        next_in_channels = self.in_channels
        for i, num_block in enumerate(num_blocks):
            output_channels = output_channels_list[i]
            stride = 1 if i == 0 else 2
            layer = self._make_layer(block, next_in_channels, output_channels, num_block, stride)
            next_in_channels = output_channels
            self.layers.append(layer)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block: Type[Union[BasicResidualBlock, BottleneckResidualBlock]],
                    input_channels: int,
                    output_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        创建ResNet层级。

        Args:
            block (Type[Union[BasicResidualBlock, BottleneckResidualBlock]]): 残差块的类型。
            input_channels (int): 输入通道数。
            output_channels (int): 输出通道数。
            num_blocks (int): 残差块的数量。
            stride (int): 第一个残差块的步幅。

        Returns:
            nn.Sequential: 包含多个残差块的层级。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        block_in_channels = input_channels
        for stride in strides:
            if block == BasicResidualBlock:
                layers.append(block(block_in_channels, output_channels, stride))
            elif block == BottleneckResidualBlock:
                process_channels = output_channels // 4
                layers.append(block(block_in_channels, output_channels, process_channels, stride))
            else:
                raise NotImplementedError
            block_in_channels = output_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, save_layer_outputs: bool = False) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为(B, C, H, W)。
            save_layer_outputs (bool, optional): 是否保存每个层级的输出。

        Returns:
            list: 每个层级的输出特征图列表。
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        outputs = []
        for layer in self.layers:
            x = layer(x)
            if save_layer_outputs:
                outputs.append(x)

        return outputs if save_layer_outputs else x


def resnet18(**kwargs: Any) -> ResNet:
    """
    创建ResNet18特征图模型。

    网络结构：
        - 输入层：接受任意大小的输入图像。
        - 卷积层1：7x7 卷积，步长为 2，通道数为 64，后接 3x3 最大池化。
        - 残差块1：2 个基本块，每个块包含 2 个通道数为 64 的 3x3 卷积层，步长为 1。
        - 残差块2：2 个基本块，通道数加倍为 128，步长为 2。
        - 残差块3：2 个基本块，通道数加倍为 256，步长为 2。
        - 残差块4：2 个基本块，通道数加倍为 512，步长为 2。

    参数量:
        不含分类头(平均值池化+全连接)，参数共`11,174,784`个，合11.17M。

    输入形状: `(B, C, H, W)`, 其中 `C` 为输入通道数(默认为 3 即 RGB)，`H` 为图像高度，`W` 为图像宽度。

    输出形状: `(B, 512, 20, 20)`。

    Args:
        **kwargs: 其他参数

    Returns:
        ResNet: ResNet18模型。
    """
    model = ResNet(
        num_blocks=[2, 2, 2, 2],
        block=BasicResidualBlock,
        **kwargs
    )
    return model


def resnet34(**kwargs: Any) -> ResNet:
    """
    创建 ResNet34 特征图模型。

    网络结构：
        - 输入层：接受任意大小的输入图像。
        - 卷积层1：7x7 卷积，步长为 2，通道数为 64，后接 3x3 最大池化。
        - 残差块1：3 个基本块，每个块包含 2 个通道数为 64 的 3x3 卷积层，步长为 1。
        - 残差块2：4 个基本块，通道数加倍为 128，步长为 2。
        - 残差块3：6 个基本块，通道数加倍为 256，步长为 2。
        - 残差块4：3 个基本块，通道数加倍为 512，步长为 2。

    参数量:
        不含分类头(平均值池化+全连接)，参数共`21,282,944`个，合21.28M。

    输入形状: `(B, C, H, W)`, 其中 `C` 为输入通道数(默认为 3 即 RGB)，`H` 为图像高度，`W` 为图像宽度。

    输出形状: `(B, 512, 20, 20)`。

    Args:
        **kwargs: 其他参数

    Returns:
        ResNet: ResNet34 模型。
    """
    model = ResNet(
        num_blocks=[3, 4, 6, 3],
        block=BasicResidualBlock,
        **kwargs
    )
    return model


def resnet50(**kwargs: Any) -> ResNet:
    """
    创建 ResNet50 特征图模型。

    网络结构：
        - 输入层：接受任意大小的输入图像。
        - 卷积层1：7x7 卷积，步长为 2，通道数为 64，后接 3x3 最大池化。
        - 残差块1：3 个瓶颈块，每个块包含 3 个卷积层 (1x1, 3x3, 1x1)，通道数为 64，步长为 1。
        - 残差块2：4 个瓶颈块，通道数加倍为 128，步长为 2。
        - 残差块3：6 个瓶颈块，通道数加倍为 256，步长为 2。
        - 残差块4：3 个瓶颈块，通道数加倍为 512，步长为 2。

    参数量:
        不含分类头(平均值池化+全连接)，参数共`23,500,416`个，合23.50M。

    输入形状: `(B, C, H, W)`, 其中 `C` 为输入通道数(默认为 3 即 RGB)，`H` 为图像高度，`W` 为图像宽度。

    输出形状: `(B, 2048, 20, 20)`。

    Args:
        **kwargs: 其他参数

    Returns:
        ResNet: ResNet50 模型。
    """
    model = ResNet(
        num_blocks=[3, 4, 6, 3],
        block=BottleneckResidualBlock,
        output_channels_list=[256, 512, 1024, 2048],
        **kwargs
    )
    return model


def resnet101(**kwargs: Any) -> ResNet:
    """
    创建 ResNet101 特征图模型。

    网络结构：
        - 输入层：接受任意大小的输入图像。
        - 卷积层1：7x7 卷积，步长为 2，通道数为 64，后接 3x3 最大池化。
        - 残差块1：3 个瓶颈块，每个块包含 3 个卷积层 (1x1, 3x3, 1x1)，通道数为 64，步长为 1。
        - 残差块2：4 个瓶颈块，通道数加倍为 128，步长为 2。
        - 残差块3：23 个瓶颈块，通道数加倍为 256，步长为 2。
        - 残差块4：3 个瓶颈块，通道数加倍为 512，步长为 2。

    参数量:
        不含分类头(平均值池化+全连接)，参数共`42,492,544`个，合42.49M。

    输入形状: `(B, C, H, W)`, 其中 `C` 为输入通道数(默认为 3 即 RGB)，`H` 为图像高度，`W` 为图像宽度。

    输出形状: `(B, 2048, 20, 20)`。

    Args:
        **kwargs: 其他参数

    Returns:
        ResNet: ResNet101 模型。
    """
    model = ResNet(
        num_blocks=[3, 4, 23, 3],
        block=BottleneckResidualBlock,
        output_channels_list=[256, 512, 1024, 2048],
        **kwargs
    )
    return model


def resnet152(**kwargs: Any) -> ResNet:
    """
    创建 ResNet152 特征图模型。

    网络结构：
        - 输入层：接受任意大小的输入图像。
        - 卷积层1：7x7 卷积，步长为 2，通道数为 64，后接 3x3 最大池化。
        - 残差块1：3 个瓶颈块，每个块包含 3 个卷积层 (1x1, 3x3, 1x1)，通道数为 64，步长为 1。
        - 残差块2：8 个瓶颈块，通道数加倍为 128，步长为 2。
        - 残差块3：36 个瓶颈块，通道数加倍为 256，步长为 2。
        - 残差块4：3 个瓶颈块，通道数加倍为 512，步长为 2。

    参数量:
        不含分类头(平均值池化+全连接)，参数共`58,136,192`个，合58.14M。

    输入形状: `(B, C, H, W)`, 其中 `C` 为输入通道数(默认为 3 即 RGB)，`H` 为图像高度，`W` 为图像宽度。

    输出形状: `(B, 2048, 20, 20)`。

    Args:
        **kwargs: 其他参数

    Returns:
        ResNet: ResNet152 模型。
    """
    model = ResNet(
        num_blocks=[3, 8, 36, 3],
        block=BottleneckResidualBlock,
        output_channels_list=[256, 512, 1024, 2048],
        **kwargs
    )
    return model
