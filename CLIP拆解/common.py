# 从 collections 模块导入 OrderedDict，用于创建有序字典，保证网络层的顺序
from collections import OrderedDict
# 导入 torch，这是 PyTorch 的主库
import torch
# 从 torch 模块导入 nn，这是构建神经网络的核心模块
from torch import nn

# --- 通用模块 ---

class LayerNorm(nn.LayerNorm):
    """
    继承自 torch.nn.LayerNorm 以处理 fp16（半精度浮点数）的计算。
    标准的 LayerNorm 在 fp16 下可能会出现数值不稳定的问题。
    """

    def forward(self, x: torch.Tensor):
        # 保存输入张量 x 的原始数据类型
        orig_type = x.dtype
        # 将输入张量 x 转换为 float32（单精度浮点数）进行 LayerNorm 计算，以保证数值稳定性
        ret = super().forward(x.type(torch.float32))
        # 将计算结果转换回原始的数据类型，以保持后续计算的一致性
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) 激活函数的一个快速近似实现。
    这种近似计算速度更快，同时效果与原始 GELU 非常接近。
    """
    def forward(self, x: torch.Tensor):
        # 应用 QuickGELU 激活函数：x * sigmoid(1.702 * x)
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    一个残差注意力块，是 Transformer 的核心构建单元。
    它包含一个多头自注意力层（Multi-head Self-Attention）和一个前馈神经网络（MLP）。
    每个部分都使用了残差连接（Residual Connection）。
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        # 调用父类 nn.Module 的构造函数
        super().__init__()

        # 定义多头注意力层
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 定义第一个 LayerNorm 层，在进入注意力层之前使用
        self.ln_1 = LayerNorm(d_model)
        # 定义一个前馈网络（MLP），由两个线性层和一个 QuickGELU 激活函数组成
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)), # 第一个线性层，将维度放大4倍
            ("gelu", QuickGELU()),                    # QuickGELU 激活函数
            ("c_proj", nn.Linear(d_model * 4, d_model)) # 第二个线性层，将维度恢复到原始大小
        ]))
        # 定义第二个 LayerNorm 层，在进入 MLP 之前使用
        self.ln_2 = LayerNorm(d_model)
        # 注意力掩码，用于防止在序列中注意到未来的位置（主要用于文本解码器）
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # 如果 attn_mask 存在，则将其移动到与输入 x 相同的设备和数据类型
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 执行多头注意力计算。Query, Key, Value 都是 x，表示是自注意力。
        # need_weights=False 表示我们不需要返回注意力权重，可以节省计算。
        # [0] 表示只取输出结果，忽略注意力权重。
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # 第一个残差连接：输入 x 加上经过 LayerNorm 和自注意力处理后的结果
        x = x + self.attention(self.ln_1(x))
        # 第二个残差连接：上一步的输出 x 加上经过 LayerNorm 和 MLP 处理后的结果
        x = x + self.mlp(self.ln_2(x))
        # 返回最终的输出张量
        return x


class Transformer(nn.Module):
    """
    Transformer 模型的主体结构。
    由多个 ResidualAttentionBlock 堆叠而成。
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 模型的特征维度
        self.width = width
        # Transformer 块的数量
        self.layers = layers
        # 使用 nn.Sequential 将多个 ResidualAttentionBlock 堆叠起来
        # 列表推导式会创建 `layers` 个 ResidualAttentionBlock 实例
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        # 直接将输入 x 传递给由多个残差注意力块组成的序列模型
        return self.resblocks(x)