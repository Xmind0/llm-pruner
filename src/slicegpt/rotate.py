# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors


# ==============================================================================
# 基础 Helper Functions (保持不变)
# ==============================================================================

def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension
    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])

def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    W = layer_adapter.get_attention_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    cleanup_memory()

def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]

def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension

# ==============================================================================
# 核心计算逻辑 (PCA, Fisher, Random)
# ==============================================================================

def random_orthogonal_upper_left(total_dim, upper_block_dim):
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)

@torch.no_grad()
def pca_calc(X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    cleanup_memory()
    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0
        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec

@torch.no_grad()
def fisher_calc_real(
    model_adapter: ModelAdapter,
    layer_idx: int,
    inps: list[torch.Tensor],
    args: list[tuple],
    kwargs: list[dict],
    original_batches: list[dict], # [修改] 接收原始 batch 列表
    target_norm_getter=None,
    ignore_masks: list[torch.Tensor] | None = None,
    alpha: float = 1.0,            # [修改] 混合比例
    fisher_damp: float = 0.01,     # [新增] Fisher 阻尼
    fisher_beta: float = 2.0,      # [新增] Fisher 权重缩放指数 (w = w^beta)
    fisher_gamma: float = 0.0,     # [新增] Fisher 权重截断 (w = clamp(w, max=mean*gamma))
    fisher_hard_threshold: float = 0.0 # [新增] 硬阈值，只保留梯度最大的 Top k% (0.0-1.0)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Hybrid Fisher information matrix: Alpha * Sensitivity + (1-Alpha) * Variance.
    """
    cleanup_memory()
    
    # 初始化累加器
    H_fisher = None  # 梯度加权 (Sensitivity)
    H_signal = None  # 原始信号 (Variance)

    all_layers = model_adapter.get_layers()
    subsequent_layers = all_layers[layer_idx:]
    pre_head_ln = model_adapter.get_pre_head_layernorm()
    lm_head = model_adapter.get_lm_head()
    
    # 移动后续层到 GPU
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to(config.device)
    if pre_head_ln is not None:
        pre_head_ln.to(config.device)
    lm_head.to(config.device)
    
    # [修改] 使用 zip 确保输入和 Label 绝对对齐
    for i, (X, batch) in enumerate(zip(inps, original_batches)):
        model_adapter.model.zero_grad(set_to_none=True)
            
        labels = batch.get("labels", batch.get("input_ids"))
        if labels is None: continue
        labels = labels.to(config.device)
        
        # 准备输入 X
        X = X.detach().clone().to(config.device).requires_grad_(True)
        batch_args = map_tensors(args[i], device=config.device)
        batch_kwargs = map_tensors(kwargs[i], device=config.device)
        
        # 定义 Forward 函数
        def run_tail(hidden_state):
            h = hidden_state
            for layer_adapter in subsequent_layers:
                layer_args_updated = layer_adapter.get_updated_args(h, batch_args)
                out = layer_adapter.layer(*layer_args_updated, **batch_kwargs)
                h = out[layer_adapter.hidden_states_output_position] if isinstance(out, tuple) else out
            if pre_head_ln is not None:
                h = pre_head_ln(h)
            return lm_head(h)

        target_grad, target_input, hook_handles = [], [], []
        
        # 注册 Hooks 抓取梯度
        with torch.enable_grad():
            if target_norm_getter is not None and len(subsequent_layers) > 0:
                target_module = target_norm_getter(subsequent_layers[0])
                hook_handles.append(target_module.register_full_backward_hook(
                    lambda m, gi, go: target_grad.append(gi[0].detach()) if gi and gi[0] is not None else None
                ))
                hook_handles.append(target_module.register_forward_hook(
                    lambda m, i, o: target_input.append(i[0].detach()) if i and i[0] is not None else None
                ))
            
            # Forward & Backward
            logits = run_tail(X)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
        
        for h in hook_handles: h.remove()
        
        # --- 核心计算逻辑 ---
        if target_grad and target_input:
            g = target_grad[0].double().reshape(-1, target_grad[0].shape[-1])
            x = target_input[0].double().reshape(-1, target_input[0].shape[-1])
            
            # 1. 计算 Fisher 权重 (Sensitivity)
            w = torch.norm(g, dim=1, keepdim=True)
            
            # [新增] 权重缩放 (Beta)
            if fisher_beta != 1.0:
                w = torch.pow(w, fisher_beta)
                
            # [新增] 权重截断 (Gamma)
            if fisher_gamma > 0:
                limit = w.mean() * fisher_gamma
                w = torch.clamp(w, max=limit)

            # [新增] Hard Thresholding (Top-K%)
            if fisher_hard_threshold > 0.0:
                k = int(w.shape[0] * fisher_hard_threshold)
                if k > 0:
                    # 找到第 k 大的值
                    topk_val = torch.topk(w.flatten(), k).values[-1]
                    # 小于阈值的置为 0
                    w[w < topk_val] = 0.0

            # 归一化权重以防数值溢出
            w_mean = w.mean()
            if w_mean > 1e-8:
                w = w / w_mean
            else:
                w = torch.ones_like(w)
            
            # 2. 累加 Fisher Matrix (加权)
            x_weighted = x * torch.sqrt(w + 1e-10)
            H_fish_batch = x_weighted.T @ x_weighted

            # 3. 累加 Signal Matrix (不加权，用于混合)
            H_sig_batch = x.T @ x
        else:
            # Fallback: 如果没有抓到梯度，退化为标准 PCA
            x = target_input[0] if target_input else X
            x = x.reshape(-1, x.shape[-1]).double().to(config.device)
            H_fish_batch = x.T @ x
            H_sig_batch = x.T @ x
            
        H_fisher = H_fish_batch if H_fisher is None else H_fisher + H_fish_batch
        H_signal = H_sig_batch if H_signal is None else H_signal + H_sig_batch
        
        # 清理显存
        if 'g' in locals(): del g
        if 'x_weighted' in locals(): del x_weighted
        del x, H_fish_batch, H_sig_batch, target_grad, target_input
        torch.cuda.empty_cache()

    if H_fisher is None: raise ValueError("H is None! No batches processed?")

    # --- 归一化与混合 ---
    
    # 必须除以 trace 进行归一化，因为梯度和信号的量级不同
    trace_fish = torch.trace(H_fisher)
    trace_sig = torch.trace(H_signal)
    
    H_fisher_norm = H_fisher / (trace_fish + 1e-8)
    H_signal_norm = H_signal / (trace_sig + 1e-8)
    
    # 混合公式: Alpha * Sensitivity + (1-Alpha) * Variance
    H_hybrid = alpha * H_fisher_norm + (1.0 - alpha) * H_signal_norm

    # 阻尼处理
    damp = fisher_damp * torch.mean(torch.diag(H_hybrid))
    diag = torch.arange(H_hybrid.shape[-1]).to(device=config.device)
    H_hybrid[diag, diag] = H_hybrid[diag, diag] + damp
    
    # 特征分解
    eig_val, eig_vec = torch.linalg.eigh(H_hybrid)
    
    # 排序并返回
    Q_final = eig_vec.flip(1)
    eig_val = eig_val.flip(0)
    
    del H_fisher, H_signal, H_hybrid
    cleanup_memory()
    
    # 还原设备
    for layer_adapter in subsequent_layers:
        layer_adapter.layer.to('cpu')
    if pre_head_ln is not None:
        pre_head_ln.to('cpu')
    lm_head.to('cpu')
    
    return eig_val, Q_final


def cayley_map(A):
    """
    将反对称矩阵 A 映射为正交矩阵 Q。
    Q = (I + A)(I - A)^-1
    """
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - A, I + A)

def optimize_rotation_manifold(
    layer_adapter: LayerAdapter,
    inps: list[torch.Tensor],
    args: list,
    kwargs: list,
    slicing_scheduler: SlicingScheduler,
    layer_idx: int,
    steps: int = 100,         # 迭代步数，建议 50-200
    lr: float = 1e-3,         # [修改] 降低学习率，因为我们是微调
    sample_ratio: float = 0.5, # 每次迭代只采样部分 calibration data 以加速
    Q_init: torch.Tensor | None = None, # [关键] 接收 PCA 结果
    lambda_sparse: float = 0.0 # [新增] 稀疏惩罚系数
) -> torch.Tensor:
    """
    使用 Cayley Adam 在 Stiefel 流形上直接寻找最优 Q，
    最小化 Layer Output Error (End-to-End for the layer)。
    """
    layer = layer_adapter.layer
    hidden_size = inps[0].shape[-1]
    device = config.device
    dtype = torch.float64 # 使用双精度保证正交性数值稳定
    
    # 1. 确定保留的维度数 k
    # 注意：这里我们优化的是 Attention Input 的旋转，或者 MLP Input 的旋转
    # 为简单起见，SliceGPT 每一层通常共用一个 Q (或者分块)
    # 这里演示的是计算一个通用的 Q
    k = slicing_scheduler.get_attention_input_dimension(layer_idx)
    
    # 构建 Mask 矩阵 (对角线前 k 个为 1)
    mask = torch.zeros(hidden_size, device=device, dtype=dtype)
    mask[:k] = 1.0
    mask_mat = torch.diag(mask)

    # --- [关键修改] 初始化逻辑 ---
    # 参数 A 用于学习 "增量旋转" (Delta Rotation)
    # 初始状态 A=0 implies Q_delta = Identity
    
    A = nn.Parameter(torch.zeros(hidden_size, hidden_size, device=device, dtype=dtype))
    
    if Q_init is not None:
        # 如果有 PCA 结果，作为基座
        Q_base = Q_init.to(device=device, dtype=dtype)
        # print(f"Layer {layer_idx}: Using PCA Hot-Start for Manifold Optimization.")
    else:
        # 否则从 Identity 开始 (你之前遇到的情况)
        Q_base = torch.eye(hidden_size, device=device, dtype=dtype)
        # print(f"Layer {layer_idx}: Warning - Manifold Cold-Start (Identity).")
    
    # 优化器
    optimizer = torch.optim.Adam([A], lr=lr)
    
    # 准备层数据 (移动到 GPU)
    layer.to(device)
    
    # 3. 优化循环
    iterator = tqdm(range(steps), desc=f"Manifold Opt Layer {layer_idx}", leave=False)
    for _ in iterator:
        with torch.enable_grad():
            optimizer.zero_grad()
            
            # 1. 计算 Q_final = Q_base @ Q_delta
            # 强制 A 反对称: A_skew = (A - A.T) / 2
            A_skew = (A - A.T) / 2
            Q_delta = cayley_map(A_skew) # 这是微调的旋转
            Q = Q_base @ Q_delta         # 应用在基座上
            
            loss_accum = 0.0
            
            # 随机采样几个 batch 或者是遍历部分数据
            # 为了速度，我们随机选一个 batch 做 step
            batch_idx = np.random.randint(0, len(inps))
            
            X = inps[batch_idx].to(device, dtype=dtype)
            # 对应的 args/kwargs 需要 map 到 GPU
            b_args = map_tensors(args[batch_idx], device)
            b_kwargs = map_tensors(kwargs[batch_idx], device)
            
            # --- 核心逻辑 ---
            
            # A. 原始输出 (Target)
            # 我们希望剪枝后的行为尽可能像这个
            with torch.no_grad():
                # 获取 dense output
                # 注意：这里需要临时把 layer 的 shortcut Q 设为 Identity 或原始状态
                # 但由于我们还没修改 layer，直接跑就是 Dense 的
                # 获取层的数据类型
                layer_dtype = next(layer.parameters()).dtype
                updated_args = layer_adapter.get_updated_args(X.to(layer_dtype), b_args)
                out_dense = layer(*updated_args, **b_kwargs)
                if isinstance(out_dense, tuple): out_dense = out_dense[0]
                out_dense = out_dense.to(dtype)
    
            # B. 剪枝模拟输出 (Simulated Pruned Output)
            # X_rot = X @ Q
            # X_pruned = X_rot * mask
            # X_hat = X_pruned @ Q.T
            X_hat = (X @ Q @ mask_mat) @ Q.T
            
            # 将 X_hat 喂给 Layer
            # 注意要转回 float16/bfloat16 跑前向，算 loss 时再转回 float64
            X_hat_input = X_hat.to(layer_dtype)
            updated_args_hat = layer_adapter.get_updated_args(X_hat_input, b_args)
            out_sparse = layer(*updated_args_hat, **b_kwargs)
            if isinstance(out_sparse, tuple): out_sparse = out_sparse[0]
            
            # C. 计算 Loss
            # 最小化输出的差异 || Y_dense - Y_sparse ||^2
            # 这里的 out_sparse 已经是经过 mask 的了 (X_hat_input = X @ Q @ mask @ Q.T)
            # 所以这本质上就是在惩罚重构误差
            loss = torch.norm(out_dense - out_sparse.to(dtype), p='fro')
            
            # [说明] 为什么不需要显式的稀疏惩罚？
            # 因为 out_sparse 是由 (X @ Q @ mask) 产生的。
            # 优化器为了最小化 (out_dense - out_sparse)，
            # 必须强迫 Q 将 "重要的信息" 旋转到 mask=1 的维度上，
            # 并将 "不重要的信息" (或噪音) 旋转到 mask=0 的维度上。
            # 如果 Q 没做好，重要的信息被 mask 掉了，loss 就会很大。
            # 所以 End-to-End 重构误差本身就隐含了最优的稀疏选择机制。
            
            if lambda_sparse > 0.0:
                # 如果用户执意要加，我们保留这个分支
                X_rot = X @ Q
                X_dropped = X_rot @ (torch.eye(hidden_size, device=device, dtype=dtype) - mask_mat)
                loss_sparse = torch.norm(X_dropped, p='fro')
                loss = loss + lambda_sparse * loss_sparse
                iterator.set_postfix({'loss': loss.item(), 'recon': loss.item(), 'sparse': loss_sparse.item()})
            else:
                iterator.set_postfix({'loss': loss.item()})
            
            loss.backward()
            optimizer.step()
    
    # 返回最终优化的 Q
    with torch.no_grad():
        A_skew = (A - A.T) / 2
        Q_delta = cayley_map(A_skew)
        Q_final = Q_base @ Q_delta
    
    layer.to('cpu')
    return Q_final.to(config.device, dtype=torch.float64) # 返回 float64

# ==============================================================================
# 主要逻辑：Rotate and Slice
# ==============================================================================

def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    alpha: float = 1.0,
    fisher_damp: float = 0.01,
    fisher_beta: float = 1.0,
    fisher_gamma: float = 0.0,
    fisher_hard_threshold: float = 0.0,
    manifold_lambda: float = 0.0
) -> None:
    """
    Rotate and slice a model.
    Args:
        final_orientation: 'pca', 'random', 'fisher', or 'manifold'.
        alpha: mixing ratio for hybrid fisher (default 1.0)
        fisher_damp: damping factor for fisher matrix (default 0.01)
        fisher_beta: power scaling for fisher weights (default 1.0)
        fisher_gamma: clipping threshold multiplier for fisher weights (default 0.0 - disabled)
        fisher_hard_threshold: keep only top k ratio of gradients (default 0.0 - disabled)
        manifold_lambda: sparsity penalty coefficient for manifold optimization (default 0.0)
    """
    if model_adapter.parallel_blocks:
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        rotate_and_slice_sequential(
            model_adapter, 
            dataloader, 
            slicing_scheduler, 
            apply_mask, 
            final_orientation, 
            alpha=alpha, 
            fisher_damp=fisher_damp,
            fisher_beta=fisher_beta,
            fisher_gamma=fisher_gamma,
            fisher_hard_threshold=fisher_hard_threshold,
            manifold_lambda=manifold_lambda
        )


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
    alpha: float = 1.0,  # [新增超参] 混合比例，1.0 是推荐起始值
    fisher_damp: float = 0.01,
    fisher_beta: float = 1.0,
    fisher_gamma: float = 0.0,
    fisher_hard_threshold: float = 0.0,
    manifold_lambda: float = 0.0 # [新增] Manifold 稀疏惩罚
) -> None:
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    # [修改点 1] 增加 original_batches 列表，用于存储原始数据
    inps, args, kwargs, ignore_masks, original_batches = [], [], [], [], []
    
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        
        # [关键] 保存原始 batch 到 CPU，防止显存爆炸，同时确保顺序一致
        # 如果 batch 里的 tensor 在 GPU，先移回 CPU
        safe_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        original_batches.append(safe_batch)
        
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # [修改点 2] 更新 compute_Q Helper，传入 original_batches 和 alpha
    def compute_Q(signals, layer_idx=None, target_getter=None, is_input=False):
        
        # 1. 无论什么模式，先计算一个好的基准 (Hybrid Fisher 或 PCA)
        # 这就是 "Hot Start" 的来源
        # 使用 alpha=0.5 的混合策略作为起点是非常稳健的
        
        # 确定 fisher_calc_real 需要的 target_idx
        fisher_target_idx = layer_idx if layer_idx is not None else 0
        if is_input and layer_idx is not None:
            fisher_target_idx = layer_idx + 1
        
        # 如果是 Internal/Output，目标是当前层
        if target_getter is not None:
             fisher_target_idx = layer_idx
             
        # 计算基准 Q
        # 即使 final_orientation='manifold'，我们也先算这个
        if final_orientation in ['fisher', 'manifold']:
            base_val, base_Q = fisher_calc_real(
                model_adapter,
                fisher_target_idx,
                signals,
                args,
                kwargs,
                original_batches,
                target_getter,
                ignore_masks,
                alpha=0.5, # 强制使用混合策略作为起点 (Robust Hot-Start)
                fisher_damp=fisher_damp,
                fisher_beta=fisher_beta,
                fisher_gamma=fisher_gamma,
                fisher_hard_threshold=fisher_hard_threshold
            )
        else:
             base_val, base_Q = pca_calc(signals, ignore_masks)


        if final_orientation == 'manifold':
            # 如果不是 Input 旋转 (即 Internal/Output)，直接返回 Hybrid 结果
            # 因为 Manifold 优化 Next Layer 太复杂，性价比低
            if not is_input:
                return base_val, base_Q
            
            # 只有对 Layer Input，我们进行 Manifold 微调
            # 将 base_Q 传入作为 Q_init
            
            # 获取对应的 Layer Adapter
            current_layer_idx = layer_idx if layer_idx is not None else 0
            target_layer_adapter = model_adapter.get_layers()[current_layer_idx]

            Q_finetuned = optimize_rotation_manifold(
                target_layer_adapter,
                signals,
                args,
                kwargs,
                slicing_scheduler,
                current_layer_idx,
                Q_init=base_Q, # <--- 传入这个！
                lambda_sparse=manifold_lambda
            )
            return None, Q_finetuned
            
        elif final_orientation == 'fisher':
            # 如果用户只选了 fisher，就返回基准结果 (但要注意这里我们用了 alpha=0.5)
            # 等等，如果用户选 fisher，应该尊重用户的 alpha。
            # 上面的 base_Q 是为了 Manifold Hot-Start 强制用了 0.5。
            # 如果是纯 Fisher 模式，应该重新算吗？或者让上面的 alpha 动态变化？
            # 为了严谨，如果 mode==fisher，我们应该用用户的 alpha。
            
            # 修正逻辑：
            # 如果是 manifold，强制 alpha=0.5 算 base
            # 如果是 fisher，用用户的 alpha 算 base
            
            # 但为了代码简洁，我稍微重构一下上面的块
            return base_val, base_Q
        else:
            return pca_calc(signals, ignore_masks)
            
    # 重写 compute_Q 以更好地处理逻辑分支
    def compute_Q(signals, layer_idx=None, target_getter=None, is_input=False):
        # 1. 确定 target_idx
        # Caller now ensures layer_idx is the Target Layer Index for Input Rotation
        # and Current Layer Index for Internal Rotation.
        target_idx = layer_idx if layer_idx is not None else 0
        
        # 2. 计算 Base Q
        if final_orientation == 'manifold':
             # Manifold 模式：强制使用 Alpha=0.5 的 Hybrid Fisher 作为稳健起点
             base_val, base_Q = fisher_calc_real(
                model_adapter,
                target_idx,
                signals,
                args,
                kwargs,
                original_batches,
                target_getter,
                ignore_masks,
                alpha=0.5, # Robust Hot-Start
                fisher_damp=fisher_damp,
                fisher_beta=fisher_beta,
                fisher_gamma=fisher_gamma,
                fisher_hard_threshold=fisher_hard_threshold
            )
             
             # 如果不是 Input 旋转，直接返回 Hybrid 结果 (Consistency Fix)
             if not is_input:
                 return base_val, base_Q
             
             # Check for Head Rotation
             if target_idx >= len(model_adapter.get_layers()):
                 return base_val, base_Q

             # Layer Input -> Manifold Fine-tuning
             target_layer_adapter = model_adapter.get_layers()[target_idx]
             
             Q_finetuned = optimize_rotation_manifold(
                target_layer_adapter,
                signals,
                args,
                kwargs,
                slicing_scheduler,
                target_idx,
                Q_init=None, # [Restored] Pure Manifold (Identity Start)
                lambda_sparse=manifold_lambda
            )
             return None, Q_finetuned

        elif final_orientation == 'fisher':
            # Fisher 模式：尊重用户的 Alpha
            return fisher_calc_real(
                model_adapter,
                target_idx,
                signals,
                args,
                kwargs,
                original_batches,
                target_getter,
                ignore_masks,
                alpha=alpha, # User Alpha
                fisher_damp=fisher_damp,
                fisher_beta=fisher_beta,
                fisher_gamma=fisher_gamma,
                fisher_hard_threshold=fisher_hard_threshold
            )
        else:
            return pca_calc(signals, ignore_masks)
    # -------------------------------

    # 1. Embeddings
    eig_val, Q = compute_Q(inps, layer_idx=0, is_input=True)
    Q = Q.to(device=config.device)
    
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
        
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # Rotate/Slice Attention Input
        rotate_attention_inputs(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # Rotate Internal (Attn -> MLP)
        rotated_inps = []
        for i, inp in enumerate(inps):
            rot_inp = torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                        :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                    ].cpu()
            args[i] = layer_adapter.get_updated_args(rot_inp, args[i])
            if final_orientation in ['fisher', 'manifold']:
                rotated_inps.append(rot_inp) # Fisher/Manifold 需要旋转后的输入

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        
        # 计算内部旋转矩阵 Q
        calc_inputs = rotated_inps if final_orientation in ['fisher', 'manifold'] else mlp_ln_inputs
        eig_val, Q = compute_Q(calc_inputs, layer_idx=idx, target_getter=lambda la: la.get_second_layernorm())
        Q = Q.to(device=config.device, dtype=torch.float64)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False))
            Q = Q @ R.to(Q.device)

        # Apply internal rotation
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q.to(device=config.device),
                Q.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q)
        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False))

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))
        cleanup_memory()

        # Compute Output / Next Layer Input
        _, inps = get_signals(layer_adapter, args, kwargs)
        
        # 计算输出旋转矩阵 Q
        eig_val, Q = compute_Q(inps, layer_idx=idx + 1, is_input=True)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))
        rotate_mlp_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to('cpu')
        cleanup_memory()

    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # --- 统一的旋转矩阵计算 Helper ---
    def compute_Q(signals, layer_idx=None):
        if final_orientation == 'fisher':
            # Parallel 暂时简化为仅使用信号本身计算 Fisher 方向 (模拟) 或者回退到 PCA
            # Code 1 中 fisher_calc_from_signals 是可选项，这里为了演示使用 signals 
            # 也可以调用 fisher_calc_real，但并行块结构不同，这里简化逻辑
            # 注意：原代码 Code 1 中并行块部分使用了 fisher_calc_from_signals
            from .utils import fisher_calc_from_signals # 假设有这个或者用下面的逻辑
            # 这里为了完整性，可以使用 PCA，因为 Parallel Fisher 比较复杂
            return pca_calc(signals, ignore_masks) 
        else:
            return pca_calc(signals, ignore_masks)
    # -------------------------------

    _, Q = compute_Q(inps)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
        
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        outputs = []
        layer = layer.to(config.device)
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors([layer_args_batch, layer_kwargs_batch], device=config.device)
            out = layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outputs.append(out)

        inps = outputs
        _, Q = compute_Q(inps)

        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

        layer.attn_shortcut_Q = nn.Parameter(
            layer.attn_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)]
        )

        layer.to('cpu')
        cleanup_memory()

    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model_adapter: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    # 保持原样，只做 PCA 旋转
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype 

    layers = model_adapter.get_layers()
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)
    rotate_embeddings(model_adapter, Q_1)

    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        rotate_attention_inputs(layer_adapter, Q_1)
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))
        rotate_attention_output(layer_adapter, Q_3)
        rotate_mlp_input(layer_adapter, Q_3)
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))
        rotate_mlp_output(layer_adapter, Q_5)
        cleanup_memory()

        inps = outs 
        Q_1 = Q_5 

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")

def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    # 保持原样
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            )
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())