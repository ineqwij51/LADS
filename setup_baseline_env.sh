#!/bin/bash

# 创建专门的conda环境用于baselines对比实验
echo "正在创建conda环境 baseline-env..."
conda create -n baseline-env python=3.9 -y

# 激活环境
echo "激活环境..."
conda activate baseline-env

# 安装PyTorch (CUDA 11.8版本，根据你的CUDA版本调整)
echo "安装PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装基础科学计算库
echo "安装基础科学计算库..."
conda install numpy scipy scikit-learn matplotlib seaborn pandas tqdm -y

# 安装tsai (用于PatchTST)
echo "安装tsai库..."
pip install tsai

# 安装timm (用于ViT)
echo "安装timm库..."
pip install timm

# 安装transformers (用于ViT备选)
echo "安装transformers库..."
pip install transformers

# 安装MMAction2 (用于PoseC3D)
echo "安装MMAction2..."
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch1.13.0/index.html
pip install mmdet
pip install mmpose
pip install mmaction2

# 安装其他可能需要的库
echo "安装其他依赖库..."
pip install einops
pip install opencv-python
pip install pillow

# 验证安装
echo "验证安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')

try:
    import tsai
    print('tsai安装成功')
except ImportError:
    print('tsai安装失败')

try:
    import timm
    print('timm安装成功')
except ImportError:
    print('timm安装失败')

try:
    import mmaction
    print('mmaction2安装成功')
except ImportError:
    print('mmaction2安装失败')

try:
    import transformers
    print('transformers安装成功')
except ImportError:
    print('transformers安装失败')
"

echo "环境设置完成！"
echo "使用方法："
echo "1. 激活环境: conda activate baseline-env"
echo "2. 运行实验: python Baselines.py"
echo ""
echo "注意事项："
echo "- 如果CUDA版本不匹配，请调整PyTorch安装命令中的CUDA版本"
echo "- 如果某些库安装失败，可能需要手动安装或使用不同的版本"
echo "- 建议在运行前测试每个模型是否能正常导入" 