# 导入 OrderedDict 用于创建有序字典
from collections import OrderedDict
# 导入 torch 主库
import torch
# 导入 nn 模块，用于构建神经网络
from torch import nn
# 导入 numpy，用于一些数值计算（如此处的 np.log）
import numpy as np

# 从我们自己创建的 common.py 文件中导入 Transformer 和 LayerNorm 模块
from common import Transformer, LayerNorm


# --- 独立的文本编码器类 ---

class TextEncoder(nn.Module):
    """CLIP 的独立文本编码器。"""

    def __init__(self, context_length: int, vocab_size: int, transformer_width: int, transformer_heads: int,
                 transformer_layers: int, embed_dim: int):
        # 调用父类构造函数
        super().__init__()

        # 文本序列的最大长度
        self.context_length = context_length
        # Transformer 主体
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # 创建并传入因果注意力掩码
        )
        # 词汇表大小
        self.vocab_size = vocab_size
        # Token 嵌入层，将 token ID 映射为向量
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # 可学习的位置编码
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # Transformer 后的最终 LayerNorm
        self.ln_final = LayerNorm(transformer_width)
        # 最终的线性投影层，将 Transformer 输出的维度映射到与图像特征一致的 embed_dim
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # 初始化模型参数
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        初始化模型的权重。这些初始化方法遵循原始 CLIP 论文的设置。
        """
        # 对 token 嵌入权重进行正态分布初始化
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # 对位置编码进行正态分布初始化
        nn.init.normal_(self.positional_embedding, std=0.01)

        # 计算 Transformer 块中不同部分权重的初始化标准差
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        # 遍历 Transformer 中的所有残差块并初始化它们的权重
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 初始化文本投影层的权重
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        构建一个因果注意力掩码 (causal attention mask)。
        这个掩码确保在预测当前 token 时，只能注意到它之前的 tokens，而不能注意到未来的 tokens。
        """
        # 创建一个方阵，大小为 (context_length, context_length)
        mask = torch.empty(self.context_length, self.context_length)
        # 用负无穷填充整个矩阵
        mask.fill_(float("-inf"))
        # 将上三角部分（包括对角线）设置为0，其余部分保持负无穷
        mask.triu_(1)  # 1 表示不包括对角线
        return mask

    @property
    def dtype(self):
        # 定义一个属性，方便获取模型的数据类型（例如 float16 或 float32）
        return self.token_embedding.weight.dtype

    def forward(self, text):
        # 1. 获取 token 嵌入并转换为正确的 dtype
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # 2. 加上位置编码
        x = x + self.positional_embedding.type(self.dtype)
        # 3. 调整维度顺序以匹配 PyTorch Transformer 的输入格式 (NLD -> LND)
        x = x.permute(1, 0, 2)
        # 4. 通过 Transformer
        x = self.transformer(x)
        # 5. 恢复维度顺序 (LND -> NLD)
        x = x.permute(1, 0, 2)
        # 6. 通过最后的 LayerNorm
        x = self.ln_final(x).type(self.dtype)

        # 7. 提取序列中最后一个 token (EOT, End of Text) 对应的特征
        # text.argmax(dim=-1) 找到每行中 EOT token 的索引
        # torch.arange(x.shape[0]) 生成批次索引
        # 通过高级索引，我们为批次中的每个序列提取其 EOT token 的输出向量
        # 然后通过 text_projection 层进行最终的线性变换
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


# --- 权重转换和模型构建函数 ---
def convert_weights(model: nn.Module):
    """一个辅助函数，用于将模型中特定层的权重和偏置转换为半精度浮点数 (fp16)。(与 image_encoder.py 中的函数相同)"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_text_encoder(state_dict: dict):
    """
    根据 CLIP 的 state_dict 构建独立的文本编码器。
    """
    # 从权重字典中推断模型的超参数
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # 使用推断的超参数初始化 TextEncoder 模型
    model = TextEncoder(
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        embed_dim=embed_dim
    )

    # 创建一个新的有序字典来存储只属于文本部分的权重
    text_state_dict = OrderedDict()
    # 定义一个列表，包含所有文本编码器独有的权重键名
    text_keys = [
        "token_embedding.weight",
        "positional_embedding",
        "text_projection",
        "ln_final.weight",
        "ln_final.bias"
    ]
    # 遍历完整的 state_dict
    for key, value in state_dict.items():
        # 如果键在我们的列表中，或者是以 "transformer.resblocks" 开头，则认为是文本模型的权重
        if key in text_keys or key.startswith("transformer.resblocks"):
            text_state_dict[key] = value

    # 将模型权重转换为 fp16
    convert_weights(model)
    # 加载筛选出的文本模型权重
    model.load_state_dict(text_state_dict)
    # 返回设置为评估模式（.eval()）的模型
    return model.eval()