import torch
import clip
from PIL import Image

# 从我们创建的文件中导入构建函数
from image_encoder import build_image_encoder
from text_encoder import build_text_encoder

# 确定设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型 "ViT-L/14@336px"
MODEL_NAME = "ViT-L/14@336px"



# 1. 加载OpenAI官方的CLIP模型以下载权重
print(f"正在从OpenAI下载原始CLIP模型和权重: {MODEL_NAME}...")
# clip.load() 会自动处理模型的下载和缓存
original_model, preprocess = clip.load(MODEL_NAME, device=device)
clip_state_dict = original_model.state_dict()

# --------------------

# 2. 使用我们分离的函数构建独立的编码器
#    我们的构建函数会自动从 state_dict 推断出 ViT-L 的架构
print("正在构建独立的图像编码器...")
image_encoder = build_image_encoder(clip_state_dict).to(device)

print("正在构建独立的文本编码器...")
text_encoder = build_text_encoder(clip_state_dict).to(device)

# 打印模型信息以验证
print(f"\n成功加载图像编码器: {image_encoder.__class__.__name__}")
print(f"图像输入分辨率: {image_encoder.input_resolution}")
if isinstance(image_encoder, torch.nn.Module):  # 避免在 VisionTransformer 之外的类型上查找不存在的属性
    if hasattr(image_encoder, 'conv1'):
        print(f"Patch Size: {image_encoder.conv1.kernel_size[0]}")
    if hasattr(image_encoder, 'transformer'):
        print(f"Transformer 层数: {len(image_encoder.transformer.resblocks)}")
        print(f"Transformer 宽度: {image_encoder.transformer.width}")

# 3. 准备输入数据
#    图像输入: preprocess 对象会根据 "ViT-L/14@336px" 模型自动调整图像大小为 336x336
#    我们使用 clip 自带的下载器下载一张示例图片
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text_inputs = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
print(type(image))
print(type(text_inputs))



# 4. 使用独立的编码器进行推理
with torch.no_grad():
    print("\n使用独立编码器进行编码...")
    image_features = image_encoder(image)
    text_features = text_encoder(text_inputs)

    # 打印输出特征的形状 (ViT-L 的输出维度是 768)
    print("图像特征形状:", image_features.shape)
    print("文本特征形状:", text_features.shape)

    # 验证：计算图文相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # 与 clip.tokenize(...) 中的顺序保持一致
    labels = ["a diagram", "a dog", "a cat"]
    # 显示 top-k，这里 k 设为文本数量（也可以改为 1/2/3）
    k = len(labels)
    values, indices = similarity[0].topk(k)

    print("\n图文相似度计算结果:")
    for value, idx in zip(values, indices):
        print(f"'{labels[idx.item()]}': {100 * value.item():.2f}%")