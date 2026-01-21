import torch
import sys
import os
import argparse

# 将 src 添加到路径，以便导入 slicegpt
sys.path.append(os.path.abspath("src"))

from slicegpt import hf_utils, layernorm_fusion, rotate, data_utils, gpu_utils
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler

# 1. 设置
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to load")
parser.add_argument("--sparsity", type=float, default=0.25, help="Sparsity level")
parser.add_argument("--cal-dataset", type=str, default="wikitext2", help="Calibration dataset")
parser.add_argument("--orientation", type=str, default="pca", choices=["pca", "random", "fisher"],
                    help="Rotation matrix calculation method: pca (default), random, or fisher")
args = parser.parse_args()

model_name = args.model
sparsity = args.sparsity
device = "cuda" if torch.cuda.is_available() else "cpu"
config.device = torch.device(device)
config.dtype = torch.float16 # 使用 fp16

print(f"正在 {device} 上加载 {model_name} ...")
print(f"使用旋转矩阵计算方法: {args.orientation}")

# 2. 加载模型和分词器
# get_model_and_tokenizer 返回通用适配器 ModelAdapter 和 HF tokenizer
model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
    model_name, 
    token=None, 
    dtype=config.dtype
)
# model_adapter.model.to(device)

# 3. 准备校准数据 (Wikitext2)
# 需要少量数据来计算压缩所需的旋转矩阵 (PCA)
print(f"正在加载校准数据 {args.cal_dataset}...")
dataset = data_utils.get_dataset(args.cal_dataset)
train_loader = data_utils.prepare_dataloader(
    dataset=dataset["train"],
    tokenizer=tokenizer,
    max_seqlen=2048,
    batch_size=1,
    nsamples=128 # 128 个样本通常足够进行校准
)

# 4. 融合 LayerNorms (旋转所需)
# 替换标准层为 SliceGPT 兼容层，并将 LayerNorm 权重融合到相邻的线性层中
print("正在融合 LayerNorms...")
layernorm_fusion.replace_layers(model_adapter)
layernorm_fusion.fuse_modules(model_adapter)

# 5. 计算切片计划 (Slicing Schedule)
original_dim = model_adapter.hidden_size
new_embedding_dimension = int((1 - sparsity) * original_dim)
# 确保维度可以被 8 整除以提高效率
new_embedding_dimension -= new_embedding_dimension % 8 

print(f"从 {original_dim} 切片到 -> {new_embedding_dimension}")
scheduler = ConstSlicingScheduler(new_embedding_dimension)

# 6. 旋转和切片
# 这是核心步骤：计算 PCA/Fisher，旋转模型，并删除最不重要的维度
print(f"正在使用 {args.orientation} 方法旋转和切片...")
rotate.rotate_and_slice(
    model_adapter, 
    train_loader, 
    scheduler, 
    final_orientation=args.orientation
)

print("成功！模型已完成切片。")
output_path = f"sliced_{model_name.split('/')[-1]}.pt"
torch.save(model_adapter.model.state_dict(), output_path)
print(f"模型已保存至 {output_path}")

# 7. 评估
print("正在评估困惑度...")
test_key = "test" if "test" in dataset else "validation"
test_loader = data_utils.prepare_test_dataloader(
    dataset=dataset[test_key],
    tokenizer=tokenizer,
    batch_size=8
)
ppl = gpu_utils.evaluate_ppl(model_adapter.model, model_adapter.model.config.pad_token_id, test_loader)
print(f"切片后 {args.cal_dataset} 困惑度: {ppl}")
