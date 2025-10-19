# 导入 OrderedDict 用于创建有序字典
from collections import OrderedDict
# 从 typing 模块导入 Tuple 用于类型注解
from typing import Tuple

# 导入 torch 主库
import torch
# 导入 torch.nn.functional，其中包含了一些有用的函数，如 multi_head_attention_forward
import torch.nn.functional as F
# 导入 nn 模块，用于构建神经网络
from torch import nn

# 从我们自己创建的 common.py 文件中导入 Transformer 和 LayerNorm 模块
from common import Transformer, LayerNorm


# --- 图像编码器专属模块 ---

class Bottleneck(nn.Module):
    """ResNet 中的瓶颈块（Bottleneck Block）结构。"""
    # expansion 因子，用于控制输出通道数相对于输入通道数的倍数
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        # 调用父类构造函数
        super().__init__()

        # 第一个 1x1 卷积层，用于降低通道数
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个 3x3 卷积层，用于提取特征
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # 如果步长大于1，则使用平均池化层进行下采样，否则使用 Identity 层（什么都不做）
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三个 1x1 卷积层，用于恢复通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        # 下采样模块，用于匹配残差连接的维度
        self.downsample = None
        self.stride = stride

        # 如果步长大于1或者输入输出通道数不匹配，则需要定义 downsample 层
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),  # 先用平均池化进行下采样
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),  # 再用1x1卷积调整通道
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        # 保存原始输入，用于残差连接
        identity = x

        # 主路径计算
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        # 如果定义了 downsample 层，则对 identity (原始输入) 进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：将主路径的输出和 identity 相加
        out += identity
        # 最后通过 ReLU 激活函数
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """
    一个使用多头注意力的二维注意力池化层。
    它取代了传统 ResNet 最后的全局平均池化层。
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 定义位置编码，spacial_dim 是特征图的边长，+1 是为类别 token 预留
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 定义用于生成 Key 的线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        # 定义用于生成 Query 的线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # 定义用于生成 Value 的线性投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 定义最终输出的线性投影层
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        # 注意力头的数量
        self.num_heads = num_heads

    def forward(self, x):
        # x 的原始形状是 NCHW (批次, 通道, 高, 宽)
        # x.flatten(start_dim=2) 将 H 和 W 合并成一个维度，形状变为 NC(HW)
        # .permute(2, 0, 1) 交换维度，变为 (HW)NC，以匹配 Transformer 的输入格式 (序列长度, 批次, 特征维度)
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        # 在序列的开头拼接一个全局平均特征作为 query，相当于类别 token
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        # 添加位置编码
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        # 调用 PyTorch 底层的多头注意力前向传播函数
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,  # query 是全局平均特征，key 和 value 是所有特征
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True, training=self.training, need_weights=False
        )
        # 返回最终的池化特征，形状为 (N, D)，移除序列长度维度
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """一个修改版的 ResNet，用于 CLIP 的图像编码器。"""

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # 定义 "stem" 部分，由三个卷积层组成，取代了标准 ResNet 的单个大卷积核层
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # 定义残差层
        self._inplanes = width  # 可变的内部变量，用于记录当前层的输入通道数
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # 定义最终的注意力池化层
        embed_dim = width * 32  # ResNet 输出的特征维度
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        # 辅助函数，用于构建一个残差层（由多个 Bottleneck 块组成）
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 定义 stem 前向传播函数
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        # 将输入数据类型转换为与模型权重一致
        x = x.type(self.conv1.weight.dtype)
        # 依次通过 stem 和四个残差层
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 通过注意力池化层得到最终的图像特征
        x = self.attnpool(x)

        return x


class VisionTransformer(nn.Module):
    """视觉 Transformer 模型，用于 CLIP 的图像编码器。"""

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # 第一个卷积层，用于将图像分割成 patch 并进行线性嵌入
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # 初始化参数的缩放因子
        scale = width ** -0.5
        # 可学习的类别嵌入（class embedding），用于汇总整个图像的信息
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # 可学习的位置编码（positional embedding），用于为每个 patch 添加位置信息
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        # Transformer 前的 LayerNorm
        self.ln_pre = LayerNorm(width)

        # Transformer 主体
        self.transformer = Transformer(width, layers, heads)

        # Transformer 后的 LayerNorm
        self.ln_post = LayerNorm(width)
        # 最终的线性投影层，将特征映射到指定的输出维度
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # 输入 x 通过卷积层，将图像转换为 patch 嵌入
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # 将 patch 展平成序列
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # 调整维度顺序以匹配 Transformer 输入
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 在序列的开头拼接类别嵌入
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        # 添加位置编码
        x = x + self.positional_embedding.to(x.dtype)
        # 通过预处理 LayerNorm
        x = self.ln_pre(x)

        # 调整维度顺序以匹配 PyTorch Transformer 的输入格式 (LND: 序列长度, 批次, 维度)
        x = x.permute(1, 0, 2)
        # 通过 Transformer
        x = self.transformer(x)
        # 恢复维度顺序 (NLD: 批次, 序列长度, 维度)
        x = x.permute(1, 0, 2)

        # 仅取出类别嵌入对应的输出，并通过后处理 LayerNorm
        x = self.ln_post(x[:, 0, :])

        # 如果定义了投影层，则进行最终的线性投影
        if self.proj is not None:
            x = x @ self.proj

        return x


# --- 权重转换和模型构建函数 ---

def convert_weights(model: nn.Module):
    """一个辅助函数，用于将模型中特定层的权重和偏置转换为半精度浮点数 (fp16)。"""

    def _convert_weights_to_fp16(l):
        # 检查层是否是卷积层或线性层
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # 转换权重
            l.weight.data = l.weight.data.half()
            # 如果存在偏置，也进行转换
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        # 检查层是否是多头注意力层
        if isinstance(l, nn.MultiheadAttention):
            # 遍历所有相关的权重和偏置属性
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                # 如果属性存在，则进行转换
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # 检查层是否有 text_projection 或 proj 属性（主要用于 ViT 和 CLIP 模型）
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    # 将 _convert_weights_to_fp16 函数应用到模型的所有子模块
    model.apply(_convert_weights_to_fp16)


def build_image_encoder(state_dict: dict):
    """
    根据 CLIP 的 state_dict 构建独立的图像编码器。
    它会自动检测模型是 ViT 还是 ResNet，并加载相应的权重。
    """
    # 通过检查 "visual.proj" 是否在权重字典中来判断模型是 ViT 还是 ResNet
    vit = "visual.proj" in state_dict

    if vit:
        # --- 如果是 Vision Transformer (ViT) ---
        # 从权重形状推断模型超参数
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        embed_dim = state_dict["text_projection"].shape[1]
        vision_heads = vision_width // 64

        # 使用推断出的超参数初始化 VisionTransformer 模型
        model = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
    else:
        # --- 如果是 ModifiedResNet ---
        # 从权重形状推断模型超参数
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
        embed_dim = state_dict["text_projection"].shape[1]
        vision_heads = vision_width * 32 // 64

        # 使用推断出的超参数初始化 ModifiedResNet 模型
        model = ModifiedResNet(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width
        )

    # 创建一个新的有序字典来存储只属于视觉部分的权重
    visual_state_dict = OrderedDict()
    # 遍历完整的 state_dict
    for key, value in state_dict.items():
        # 如果键以 "visual." 开头，说明是视觉模型的权重
        if key.startswith("visual."):
            # 将 "visual." 前缀去掉，然后存入新的字典中
            visual_state_dict[key.replace("visual.", "", 1)] = value

    # 将模型权重转换为 fp16
    convert_weights(model)
    # 加载筛选出的视觉模型权重
    model.load_state_dict(visual_state_dict)
    # 返回设置为评估模式（.eval()）的模型
    return model.eval()