import torch
import numpy as np
from torch_geometric.data import Data, Dataset, Batch
from captum.attr import IntegratedGradients


def node_importance(func_graphs, struct_graphs, t1, asl, model, loader, device):
    model.eval()
    all_func = []
    all_struct = []
    all_func_delta = []
    all_struct_delta = []

    for batch_idx in loader:
        bf = Batch.from_data_list([func_graphs[i] for i in batch_idx]).to(device)
        bs = Batch.from_data_list([struct_graphs[i] for i in batch_idx]).to(device)
        t1 = t1[batch_idx].to(device)
        asl = asl[batch_idx].to(device)

        # 节点级预测
        with torch.no_grad():
            logits = model.forward(
                bf.x, bf.edge_index, bf.edge_attr,
                bs.x, bs.edge_index, bs.edge_attr,
                bf.y, bf.batch, t1, asl, mode='eval'
            )
            pred_classes = logits.argmax(dim=1)  # [batch]

        model.set_graph(bf.edge_index, bf.edge_attr,
                        bs.edge_index, bs.edge_attr,
                        bf.batch, t1, asl)

        ig = IntegratedGradients(model.forward_for_captum)
        baseline = torch.mean(bf.x, dim=0, keepdim=True)

        func_attr, func_delta = ig.attribute(
            inputs=bf.x,
            baselines=baseline,
            n_steps=10,
            target=1,
            return_convergence_delta=True,
            additional_forward_args=(
                bs.x, bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bf.y, bf.batch, t1, asl, 'func'
            )
        )
        baseline = torch.mean(bs.x, dim=0, keepdim=True)
        struct_attr, struct_delta = ig.attribute(
            inputs=bs.x,
            baselines=baseline,
            n_steps=10,
            target=1,
            return_convergence_delta=True,
            additional_forward_args=(
                bf.x, bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bf.y, bf.batch, t1, asl, 'struct'
            )
        )

        func_attr = func_attr.view(-1, 90, 2)[:, :, 1]  # 类别1
        struct_attr = struct_attr.view(-1, 90, 2)[:, :, 1]

        mask = (pred_classes == 1)
        func_attr_masked = func_attr[mask]
        struct_attr_masked = struct_attr[mask]

        all_func.append(func_attr_masked)
        all_struct.append(struct_attr_masked)
        all_func_delta.append(func_delta.abs())
        all_struct_delta.append(struct_delta.abs())

    # 合并所有 batch
    all_func = torch.cat(all_func, dim=0)  # [num_selected_samples, num_nodes]
    all_struct = torch.cat(all_struct, dim=0)
    node_importance_func = all_func.mean(dim=0)
    node_importance_struct = all_struct.mean(dim=0)
    avg_func_delta = torch.cat(all_func_delta).mean().item()
    avg_struct_delta = torch.cat(all_struct_delta).mean().item()

    print('func_delta', avg_func_delta)
    print('struct_delta', avg_struct_delta)

    # === 归一化 ===
    node_importance_func_norm = (node_importance_func - node_importance_func.min()) / (
        node_importance_func.max() - node_importance_func.min() + 1e-8
    )
    node_importance_struct_norm = (node_importance_struct - node_importance_struct.min()) / (
        node_importance_struct.max() - node_importance_struct.min() + 1e-8
    )

    #一致性度量
    cos_sim_func_struct = torch.nn.functional.cosine_similarity(
        node_importance_func_norm, node_importance_struct_norm, dim=0
    ).item()

    #一致性脑区
    node_importance_mean = (node_importance_func_norm + node_importance_struct_norm) / 2

    return node_importance_func_norm, node_importance_struct_norm, node_importance_mean, avg_func_delta, avg_struct_delta


@torch.no_grad()
def node_importance_node_level(fg_test, sg_test, Xt_test, Xa_test, model, loader_test, device,
                                cn_baseline_func=None, cn_baseline_struct=None):
    """
    ✅ 修复版：计算节点级别的贡献度

    主要改进：
    1. 使用CN均值作为baseline（如果提供）
    2. 固定target=1（AD类）
    3. 不取绝对值，保留符号信息
    4. 改进归一化：先平均再归一化
    5. 增加收敛性检查

    Args:
        fg_test, sg_test: 功能/结构图列表
        Xt_test, Xa_test: T1/ASL 数据（可以为 None）
        model: 模型
        loader_test: 数据加载器
        device: 设备
        cn_baseline_func: CN组功能模态均值 [1, feature_dim]（可选）
        cn_baseline_struct: CN组结构模态均值 [1, feature_dim]（可选）

    Returns:
        node_importance_func_final: 功能模态节点重要性 [90]
        node_importance_struct_final: 结构模态节点重要性 [90]
        node_importance_mean_final: 平均节点重要性 [90]
    """
    model.eval()
    ig = IntegratedGradients(model.forward_for_captum_node)
    N = 90
    n_steps = 10

    # ===== 修复1: 保存原始值，不在batch内归一化 =====
    all_func_raw = []
    all_struct_raw = []
    convergence_deltas_func = []
    convergence_deltas_struct = []

    for batch_data in loader_test:
        bf = Batch.from_data_list([fg_test[i] for i in batch_data]).to(device)
        bs = Batch.from_data_list([sg_test[i] for i in batch_data]).to(device)

        # 处理 None 的情况
        if Xt_test is not None and Xa_test is not None:
            t1_batch = Xt_test[batch_data].to(device)
            asl_batch = Xa_test[batch_data].to(device)
        else:
            t1_batch = None
            asl_batch = None

        model.set_graph(bf.edge_index, bf.edge_attr,
                        bs.edge_index, bs.edge_attr,
                        bf.batch, t1_batch, asl_batch)

        # ===== 修复2: 使用CN均值作为baseline =====
        if cn_baseline_func is not None:
            baseline_func = cn_baseline_func.expand_as(bf.x)
        else:
            # 如果没有预先计算，使用batch均值
            baseline_func = bf.x.mean(dim=0, keepdim=True).expand_as(bf.x)

        if cn_baseline_struct is not None:
            baseline_struct = cn_baseline_struct.expand_as(bs.x)
        else:
            baseline_struct = bs.x.mean(dim=0, keepdim=True).expand_as(bs.x)

        # ===== 修复3: 固定target=1（AD类） =====
        num_graphs = bf.batch.max().item() + 1
        target = torch.ones(num_graphs * N, dtype=torch.long).to(device)

        # ===== 功能模态 IG =====
        attr_func, delta_func = ig.attribute(
            inputs=bf.x,
            baselines=baseline_func,
            n_steps=n_steps,  # ✅ 增加步数提高精度
            target=target,
            additional_forward_args=(
                bs.x, bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bf.y, bf.batch, t1_batch, asl_batch, 'func'
            ),
            return_convergence_delta=True
        )

        # ===== 修复4: 不取绝对值，保留符号 =====
        node_importance_func = attr_func.mean(dim=1)  # [B*90]
        node_importance_func = node_importance_func.view(num_graphs, N)  # [B, 90]

        # 保存原始值（不归一化）
        all_func_raw.append(node_importance_func)
        convergence_deltas_func.append(delta_func.abs().mean().item())

        # ===== 结构模态 IG =====
        attr_struct, delta_struct = ig.attribute(
            inputs=bs.x,
            baselines=baseline_struct,
            n_steps=n_steps,
            target=target,
            additional_forward_args=(
                bf.x, bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bs.y, bs.batch, t1_batch, asl_batch, 'struct'
            ),
            return_convergence_delta=True
        )

        node_importance_struct = attr_struct.mean(dim=1)
        node_importance_struct = node_importance_struct.view(num_graphs, N)

        all_struct_raw.append(node_importance_struct)
        convergence_deltas_struct.append(delta_struct.abs().mean().item())

    # ===== 收敛性检查 =====
    avg_delta_func = np.mean(convergence_deltas_func)
    avg_delta_struct = np.mean(convergence_deltas_struct)
    print(f"Convergence delta - Func: {avg_delta_func:.6f}, Struct: {avg_delta_struct:.6f}")
    if avg_delta_func > 0.01 or avg_delta_struct > 0.01:
        print("Warning: Large convergence delta detected. Consider increasing n_steps.")

    # ===== 修复5: 先跨样本平均，再归一化 =====
    all_func_concat = torch.cat(all_func_raw, dim=0)  # [total_samples, 90]
    all_struct_concat = torch.cat(all_struct_raw, dim=0)

    # 跨样本平均（保留原始量级）
    node_importance_func_avg = all_func_concat.mean(dim=0)  # [90]
    node_importance_struct_avg = all_struct_concat.mean(dim=0)

    # 最后统一归一化（Min-Max到[0,1]）
    node_importance_func_final = normalize_final(node_importance_func_avg)
    node_importance_struct_final = normalize_final(node_importance_struct_avg)

    # 融合
    node_importance_mean_final = (node_importance_func_final + node_importance_struct_final) / 2

    return node_importance_func_final, node_importance_struct_final, node_importance_mean_final


@torch.no_grad()
def node_importance_node_level_new(fg_test, sg_test, Xt_test, Xa_test, model, loader_test, device,
                                cn_baseline_func=None, cn_baseline_struct=None, cn_baseline_t1=None, cn_baseline_asl=None):
    """
    ✅ 修复版：计算节点级别的贡献度

    主要改进：
    1. 使用CN均值作为baseline（如果提供）
    2. 固定target=1（AD类）
    3. 不取绝对值，保留符号信息
    4. 改进归一化：先平均再归一化
    5. 增加收敛性检查

    Args:
        fg_test, sg_test: 功能/结构图列表
        Xt_test, Xa_test: T1/ASL 数据（可以为 None）
        model: 模型
        loader_test: 数据加载器
        device: 设备
        cn_baseline_func: CN组功能模态均值 [1, feature_dim]（可选）
        cn_baseline_struct: CN组结构模态均值 [1, feature_dim]（可选）

    Returns:
        node_importance_func_final: 功能模态节点重要性 [90]
        node_importance_struct_final: 结构模态节点重要性 [90]
        node_importance_mean_final: 平均节点重要性 [90]
    """
    model.eval()
    ig = IntegratedGradients(model.forward_for_captum_node)
    N = 90
    n_steps = 10

    # ===== 修复1: 保存原始值，不在batch内归一化 =====
    all_func_raw = []
    all_struct_raw = []
    convergence_deltas_func = []
    convergence_deltas_struct = []

    all_t1_raw = []
    all_asl_raw = []
    convergence_deltas_t1 = []
    convergence_deltas_asl = []

    for batch_data in loader_test:
        bf = Batch.from_data_list([fg_test[i] for i in batch_data]).to(device)
        bs = Batch.from_data_list([sg_test[i] for i in batch_data]).to(device)

        # 处理 None 的情况
        if Xt_test is not None and Xa_test is not None:
            t1_batch = Xt_test[batch_data].to(device)
            asl_batch = Xa_test[batch_data].to(device)
        else:
            t1_batch = None
            asl_batch = None

        bs_x = bs.x.clone().detach().requires_grad_(True)
        bf_x = bf.x.clone().detach().requires_grad_(True)
        t1_batch = t1_batch.clone().detach().requires_grad_(True)   # shape 应与模型期望一致 (B, N, feat) 或 (B*N, feat) 按实现调整
        asl_batch = asl_batch.clone().detach().requires_grad_(True)


        model.set_graph(bf.edge_index, bf.edge_attr,
                        bs.edge_index, bs.edge_attr,
                        bf.batch, t1_batch, asl_batch)

        # ===== 修复2: 使用CN均值作为baseline =====
        if cn_baseline_func is not None:
            baseline_func = cn_baseline_func.expand_as(bf.x)
        else:
            # 如果没有预先计算，使用batch均值
            baseline_func = bf.x.mean(dim=0, keepdim=True).expand_as(bf.x)

        if cn_baseline_struct is not None:
            baseline_struct = cn_baseline_struct.expand_as(bs.x)
        else:
            baseline_struct = bs.x.mean(dim=0, keepdim=True).expand_as(bs.x)

        if cn_baseline_t1 is not None:
            baseline_t1 = cn_baseline_t1.expand_as(t1_batch)
        else:
            # 如果没有预先计算，使用batch均值
            baseline_t1 = t1_batch.mean(dim=0, keepdim=True).expand_as(t1_batch)

        if cn_baseline_asl is not None:
            baseline_asl = cn_baseline_asl.expand_as(asl_batch)
        else:
            # 如果没有预先计算，使用batch均值
            baseline_asl = asl_batch.mean(dim=0, keepdim=True).expand_as(asl_batch)

        if t1_batch is not None:
            # 若是 (B*N, feat) -> 重塑为 (B, N, feat)
            if t1_batch.dim() == 2 and t1_batch.shape[0] == num_graphs * N:
                t1_in = t1_batch.view(num_graphs, N, -1)
            else:
                t1_in = t1_batch
            t1_batch = t1_in.clone().detach().requires_grad_(True)
            # baseline 可用 CN 均值或 batch 均值，shape 要匹配 (B, N, feat)
            baseline_t1 = (cn_baseline_t1.expand_as(t1_in) if 'cn_baseline_t1' in locals() else t1_in.mean(dim=0, keepdim=True).expand_as(t1_in))
        else:
            t1_batch = None
            baseline_t1 = None

        # 处理 asl（同上）
        if asl_batch is not None:
            if asl_batch.dim() == 2 and asl_batch.shape[0] == num_graphs * N:
                asl_in = asl_batch.view(num_graphs, N, -1)
            else:
                asl_in = asl_batch
            asl_batch = asl_in.clone().detach().requires_grad_(True)
            baseline_asl = (cn_baseline_asl.expand_as(asl_in) if 'cn_baseline_asl' in locals() else asl_in.mean(dim=0, keepdim=True).expand_as(asl_in))
        else:
            asl_batch = None
            baseline_asl = None

        # ===== 修复3: 固定target=1（AD类） =====
        num_graphs = bf.batch.max().item() + 1
        target = torch.ones(num_graphs * N, dtype=torch.long).to(device)

        with torch.enable_grad():
            ig_joint = IntegratedGradients(model.forward_for_captum_with_t1_asl)

            inputs_joint = (bf_x, bs_x, t1_batch, asl_batch)

            baselines_joint = (
                baseline_func, baseline_struct,
                baseline_t1, baseline_asl
            )

            attr_tuple, delta_joint = ig_joint.attribute(
                inputs=inputs_joint,
                baselines=baselines_joint,
                n_steps=200,           # 可调，若 delta 大再增
                target=target,              # 标量类别索引
                additional_forward_args=(
                    bf.edge_index, bf.edge_attr,
                    bs.edge_index, bs.edge_attr,
                    bf.y, bf.batch, 'joint'
                ),
                return_convergence_delta=True
            )

        # 处理 t1/asl attribution（聚合到每个节点的标量贡献）
        attr_func = attr_tuple[0]  # 可能形状 (B, N, feat) 或 (B*N, feat)
        if attr_func is not None:
            if attr_func.dim() == 3:
                func_node_imp = attr_func.mean(dim=-1)               # (B, N)
            else:
                func_node_imp = attr_func.mean(dim=1).view(num_graphs, N)
            all_func_raw.append(func_node_imp)
            # convergence_deltas_func.append(delta_joint[0].abs().mean().item())

        attr_struct = attr_tuple[1]
        if attr_struct is not None:
            if attr_struct.dim() == 3:
                struct_node_imp = attr_struct.mean(dim=-1)
            else:
                struct_node_imp = attr_struct.mean(dim=1).view(num_graphs, N)
            all_struct_raw.append(struct_node_imp)
            # convergence_deltas_struct.append(delta_joint[1].abs().mean().item())

        attr_t1 = attr_tuple[2]  # 可能形状 (B, N, feat) 或 (B*N, feat)
        if attr_t1 is not None:
            if attr_t1.dim() == 3:
                t1_node_imp = attr_t1.mean(dim=-1)               # (B, N)
            else:
                t1_node_imp = attr_t1.mean(dim=1).view(num_graphs, N)
            all_t1_raw.append(t1_node_imp)
            # convergence_deltas_t1.append(delta_joint[2].abs().mean().item())

        attr_asl = attr_tuple[3]
        if attr_asl is not None:
            if attr_asl.dim() == 3:
                asl_node_imp = attr_asl.mean(dim=-1)
            else:
                asl_node_imp = attr_asl.mean(dim=1).view(num_graphs, N)
            all_asl_raw.append(asl_node_imp)
            # convergence_deltas_asl.append(delta_joint[3].abs().mean().item())



    # 在返回前 detach()
    node_importance_func_final = normalize_final(torch.cat(all_func_raw, dim=0).mean(dim=0)).detach() \
        if len(all_func_raw)>0 else None
    node_importance_struct_final = normalize_final(torch.cat(all_struct_raw, dim=0).mean(dim=0)).detach() \
        if len(all_struct_raw)>0 else None
    node_importance_t1_final = normalize_final(torch.cat(all_t1_raw, dim=0).mean(dim=0)).detach() \
        if len(all_t1_raw)>0 else None
    node_importance_asl_final = normalize_final(torch.cat(all_asl_raw, dim=0).mean(dim=0)).detach() \
        if len(all_asl_raw)>0 else None
    node_importance_mean_final = (node_importance_func_final + node_importance_struct_final + node_importance_t1_final + node_importance_asl_final) / 4

    return node_importance_func_final, node_importance_struct_final, node_importance_t1_final, node_importance_asl_final, node_importance_mean_final
    #     # ===== 功能模态 IG =====
    #     attr_func, delta_func = ig.attribute(
    #         inputs=bf.x,
    #         baselines=baseline_func,
    #         n_steps=n_steps,  # ✅ 增加步数提高精度
    #         target=target,
    #         additional_forward_args=(
    #             bs.x, bf.edge_index, bf.edge_attr,
    #             bs.edge_index, bs.edge_attr,
    #             bf.y, bf.batch, t1_batch, asl_batch, 'func'
    #         ),
    #         return_convergence_delta=True
    #     )

    #     # ===== 修复4: 不取绝对值，保留符号 =====
    #     node_importance_func = attr_func.mean(dim=1)  # [B*90]
    #     node_importance_func = node_importance_func.view(num_graphs, N)  # [B, 90]

    #     # 保存原始值（不归一化）
    #     all_func_raw.append(node_importance_func)
    #     convergence_deltas_func.append(delta_func.abs().mean().item())

    #     # ===== 结构模态 IG =====
    #     attr_struct, delta_struct = ig.attribute(
    #         inputs=bs.x,
    #         baselines=baseline_struct,
    #         n_steps=n_steps,
    #         target=target,
    #         additional_forward_args=(
    #             bf.x, bf.edge_index, bf.edge_attr,
    #             bs.edge_index, bs.edge_attr,
    #             bs.y, bs.batch, t1_batch, asl_batch, 'struct'
    #         ),
    #         return_convergence_delta=True
    #     )

    #     node_importance_struct = attr_struct.mean(dim=1)
    #     node_importance_struct = node_importance_struct.view(num_graphs, N)

    #     all_struct_raw.append(node_importance_struct)
    #     convergence_deltas_struct.append(delta_struct.abs().mean().item())

    # # ===== 收敛性检查 =====
    # avg_delta_func = np.mean(convergence_deltas_func)
    # avg_delta_struct = np.mean(convergence_deltas_struct)
    # print(f"Convergence delta - Func: {avg_delta_func:.6f}, Struct: {avg_delta_struct:.6f}")
    # if avg_delta_func > 0.01 or avg_delta_struct > 0.01:
    #     print("⚠️ Warning: Large convergence delta detected. Consider increasing n_steps.")

    # # ===== 修复5: 先跨样本平均，再归一化 =====
    # all_func_concat = torch.cat(all_func_raw, dim=0)  # [total_samples, 90]
    # all_struct_concat = torch.cat(all_struct_raw, dim=0)

    # # 跨样本平均（保留原始量级）
    # node_importance_func_avg = all_func_concat.mean(dim=0)  # [90]
    # node_importance_struct_avg = all_struct_concat.mean(dim=0)

    # # 最后统一归一化（Min-Max到[0,1]）
    # node_importance_func_final = normalize_final(node_importance_func_avg)
    # node_importance_struct_final = normalize_final(node_importance_struct_avg)

    # # 融合
    # node_importance_mean_final = (node_importance_func_final + node_importance_struct_final) / 2

    # return node_importance_func_final, node_importance_struct_final, node_importance_mean_final


@torch.no_grad()
def node_importance_node_level_SS(fg_test, sg_test, Xt_test, Xa_test, model, loader_test, device,
                                cn_baseline_func=None, cn_baseline_struct=None):
    """
    ✅ 修复版：计算节点级别的贡献度

    主要改进：
    1. 使用CN均值作为baseline（如果提供）
    2. 固定target=1（AD类）
    3. 不取绝对值，保留符号信息
    4. 改进归一化：先平均再归一化
    5. 增加收敛性检查

    Args:
        fg_test, sg_test: 功能/结构图列表
        Xt_test, Xa_test: T1/ASL 数据（可以为 None）
        model: 模型
        loader_test: 数据加载器
        device: 设备
        cn_baseline_func: CN组功能模态均值 [1, feature_dim]（可选）
        cn_baseline_struct: CN组结构模态均值 [1, feature_dim]（可选）

    Returns:
        node_importance_func_final: 功能模态节点重要性 [90]
        node_importance_struct_final: 结构模态节点重要性 [90]
        node_importance_mean_final: 平均节点重要性 [90]
    """
    model.eval()
    ig_func_sh = IntegratedGradients(model.forward_func_shared_for_ig)
    ig_func_sp = IntegratedGradients(model.forward_func_spe_for_ig)
    ig_struct_sh = IntegratedGradients(model.forward_struct_shared_for_ig)
    ig_struct_sp = IntegratedGradients(model.forward_struct_spe_for_ig)
    N = 90
    n_steps = 200

    # ===== 修复1: 保存原始值，不在batch内归一化 =====
    all_func_sh_raw = []
    all_func_sp_raw = []
    all_struct_sh_raw = []
    all_struct_sp_raw = []
    convergence_deltas_func_sh = []
    convergence_deltas_func_sp = []
    convergence_deltas_struct_sh = []
    convergence_deltas_struct_sp = []

    for batch_data in loader_test:
        bf = Batch.from_data_list([fg_test[i] for i in batch_data]).to(device)
        bs = Batch.from_data_list([sg_test[i] for i in batch_data]).to(device)

        # 处理 None 的情况
        if Xt_test is not None and Xa_test is not None:
            t1_batch = Xt_test[batch_data].to(device)
            asl_batch = Xa_test[batch_data].to(device)
        else:
            t1_batch = None
            asl_batch = None

        model.set_graph(bf.edge_index, bf.edge_attr,
                        bs.edge_index, bs.edge_attr,
                        bf.batch, t1_batch, asl_batch)

        # ===== 修复2: 使用CN均值作为baseline =====
        if cn_baseline_func is not None:
            baseline_func = cn_baseline_func.expand_as(bf.x)
        else:
            # 如果没有预先计算，使用batch均值
            baseline_func = bf.x.mean(dim=0, keepdim=True).expand_as(bf.x)

        if cn_baseline_struct is not None:
            baseline_struct = cn_baseline_struct.expand_as(bs.x)
        else:
            baseline_struct = bs.x.mean(dim=0, keepdim=True).expand_as(bs.x)

        bf_x = bf.x.clone().detach().requires_grad_(True)
        bs_x = bs.x.clone().detach().requires_grad_(True)

        # ===== 修复3: 固定target=1（AD类） =====
        num_graphs = bf.batch.max().item() + 1
        target = torch.ones(num_graphs * N, dtype=torch.long).to(device)
        print('target shape:', target.shape)
        print('num_graphs:', num_graphs)
        print('bf.x shape:', bf.x.shape)

        # ===== 功能模态 IG =====
        # with torch.enable_grad():
        attr_func_sh, delta_func_sh = ig_func_sh.attribute(
            inputs=bf_x,
            baselines=baseline_func,
            n_steps=n_steps,  # ✅ 增加步数提高精度
            target=target,
            additional_forward_args=(
                bf.edge_index, bf.edge_attr,
                bf.batch
            ),
            return_convergence_delta=True
        )
        attr_func_sp, delta_func_sp = ig_func_sp.attribute(
            inputs=bf_x,
            baselines=baseline_func,
            n_steps=n_steps,  # ✅ 增加步数提高精度
            target=target,
            additional_forward_args=(
                bf.edge_index, bf.edge_attr,
                bf.batch
            ),
            return_convergence_delta=True
        )

        # ===== 修复4: 不取绝对值，保留符号 =====
        node_importance_func_sh = attr_func_sh.mean(dim=1)  # [B*90]
        node_importance_func_sh = node_importance_func_sh.view(num_graphs, N)  # [B, 90]
        node_importance_func_sp = attr_func_sp.mean(dim=1)
        node_importance_func_sp = node_importance_func_sp.view(num_graphs, N)

        # 保存原始值（不归一化）
        all_func_sh_raw.append(node_importance_func_sh)
        all_func_sp_raw.append(node_importance_func_sp)
        convergence_deltas_func_sh.append(delta_func_sh.abs().mean().item())
        convergence_deltas_func_sp.append(delta_func_sp.abs().mean().item())

        # ===== 结构模态 IG =====
        # with torch.enable_grad():
        attr_struct_sh, delta_struct_sh = ig_struct_sh.attribute(
            inputs=bs_x,
            baselines=baseline_struct,
            n_steps=n_steps,
            target=target,
            additional_forward_args=(
                bs.edge_index, bs.edge_attr,
                bs.batch
            ),
            return_convergence_delta=True
        )
        attr_struct_sp, delta_struct_sp = ig_struct_sp.attribute(
            inputs=bs_x,
            baselines=baseline_struct,
            n_steps=n_steps,
            target=target,
            additional_forward_args=(
                bs.edge_index, bs.edge_attr,
                bs.batch
            ),
            return_convergence_delta=True
        )

        node_importance_struct_sh = attr_struct_sh.mean(dim=1)
        node_importance_struct_sh = node_importance_struct_sh.view(num_graphs, N)
        node_importance_struct_sp = attr_struct_sp.mean(dim=1)
        node_importance_struct_sp = node_importance_struct_sp.view(num_graphs, N)

        all_struct_sh_raw.append(node_importance_struct_sh)
        all_struct_sp_raw.append(node_importance_struct_sp)
        convergence_deltas_struct_sh.append(delta_struct_sh.abs().mean().item())
        convergence_deltas_struct_sp.append(delta_struct_sp.abs().mean().item())
    # ===== 收敛性检查 =====
    avg_delta_func_sh = np.mean(convergence_deltas_func_sh)
    avg_delta_func_sp = np.mean(convergence_deltas_func_sp)
    avg_delta_struct_sh = np.mean(convergence_deltas_struct_sh)
    avg_delta_struct_sp = np.mean(convergence_deltas_struct_sp)
    print(f"Convergence delta - Func SH: {avg_delta_func_sh:.6f}, Func SP: {avg_delta_func_sp:.6f}, Struct SH: {avg_delta_struct_sh:.6f}, Struct SP: {avg_delta_struct_sp:.6f}")
    if avg_delta_func_sh > 0.01 or avg_delta_func_sp > 0.01 or avg_delta_struct_sh > 0.01 or avg_delta_struct_sp > 0.01:
        print("⚠️ Warning: Large convergence delta detected. Consider increasing n_steps.")

    # ===== 修复5: 先跨样本平均，再归一化 =====
    all_func_concat_sh = torch.cat(all_func_sh_raw, dim=0)  # [total_samples, 90]
    all_func_concat_sp = torch.cat(all_func_sp_raw, dim=0)
    all_struct_concat_sh = torch.cat(all_struct_sh_raw, dim=0)
    all_struct_concat_sp = torch.cat(all_struct_sp_raw, dim=0)

    # 跨样本平均（保留原始量级）
    node_importance_func_avg_sh = all_func_concat_sh.mean(dim=0)  # [90]
    node_importance_func_avg_sp = all_func_concat_sp.mean(dim=0)
    node_importance_struct_avg_sh = all_struct_concat_sh.mean(dim=0)
    node_importance_struct_avg_sp = all_struct_concat_sp.mean(dim=0)

    # 最后统一归一化（Min-Max到[0,1]）
    node_importance_func_final_sh = normalize_final(node_importance_func_avg_sh)
    node_importance_func_final_sp = normalize_final(node_importance_func_avg_sp)
    node_importance_struct_final_sh = normalize_final(node_importance_struct_avg_sh)
    node_importance_struct_final_sp = normalize_final(node_importance_struct_avg_sp)


    return node_importance_func_final_sh, node_importance_func_final_sp, node_importance_struct_final_sh, node_importance_struct_final_sp

def normalize_final(attr):
    """
    最终归一化：Min-Max到[0,1]
    """
    attr_min = attr.min()
    attr_max = attr.max()

    if (attr_max - attr_min).abs() < 1e-8:
        return torch.zeros_like(attr)

    return (attr - attr_min) / (attr_max - attr_min)


def compute_cn_baseline(fg_train, sg_train, Xt_train, Xa_train, labels_train, device):
    """
    预先计算CN组的均值特征作为baseline

    Args:
        fg_train/sg_train: 训练集图列表
        labels_train: 训练集标签 [n_samples]
        device: 设备

    Returns:
        cn_baseline_func: CN组功能模态均值 [1, feature_dim]
        cn_baseline_struct: CN组结构模态均值 [1, feature_dim]
    """
    # 找到CN样本的索引
    if isinstance(labels_train, torch.Tensor):
        cn_indices = (labels_train == 0).nonzero(as_tuple=True)[0].cpu().numpy()
    else:
        cn_indices = np.where(np.array(labels_train) == 0)[0]

    print(f"Computing CN baseline from {len(cn_indices)} CN samples...")

    # 收集CN组的所有节点特征
    cn_func_features = []
    cn_struct_features = []
    cn_t1_features = []
    cn_asl_features = []

    for idx in cn_indices:
        cn_func_features.append(fg_train[idx].x)
        cn_struct_features.append(sg_train[idx].x)
        cn_t1_features.append(Xt_train[idx])
        cn_asl_features.append(Xa_train[idx])

    # 计算均值
    cn_func_mean = torch.cat(cn_func_features).mean(dim=0, keepdim=True).to(device)
    cn_struct_mean = torch.cat(cn_struct_features).mean(dim=0, keepdim=True).to(device)
    cn_t1_mean = torch.cat(cn_t1_features).mean(dim=0, keepdim=True).to(device)
    cn_asl_mean = torch.cat(cn_asl_features).mean(dim=0, keepdim=True).to(device)

    print(f"CN baseline shape - Func: {cn_func_mean.shape}, Struct: {cn_struct_mean.shape}")

    return cn_func_mean, cn_struct_mean, cn_t1_mean, cn_asl_mean

@torch.no_grad()
def node_importance_node_level_old(fg_test, sg_test, Xt_test, Xa_test, model, loader_test, device):
    """
    ⚠️ 旧版本（保留用于对比）
    """
    model.eval()
    ig = IntegratedGradients(model.forward_for_captum_node)
    N=90
    all_func, all_struct, all_mean = [], [], []

    for batch_data in loader_test:
        bf = Batch.from_data_list([fg_test[i] for i in batch_data]).to(device)
        bs = Batch.from_data_list([sg_test[i] for i in batch_data]).to(device)
        t1_batch = Xt_test[batch_data].to(device)
        asl_batch = Xa_test[batch_data].to(device)

        model.set_graph(bf.edge_index, bf.edge_attr,
                        bs.edge_index, bs.edge_attr,
                        bf.batch, t1_batch, asl_batch)

        with torch.no_grad():
            logits_nodes = model.forward_for_captum_node(
                bf.x, bs.x,
                bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bf.y, bf.batch, t1_batch, asl_batch, mode='func'
            )
            pred_classes_nodes = logits_nodes.argmax(dim=1)

        # === baseline ===
        baseline_func = torch.zeros_like(bf.x)
        baseline_struct = torch.zeros_like(bs.x)

        # === 功能模态 IG ===
        attr_func, _ = ig.attribute(
            inputs=bf.x,
            baselines=baseline_func,
            n_steps=10,
            target=pred_classes_nodes,
            additional_forward_args=(
                bs.x, bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bf.y, bf.batch, t1_batch, asl_batch, 'func'
            ),
            return_convergence_delta=True
        )
        node_importance_func = attr_func.abs().mean(dim=1)
        node_importance_func = node_importance_func.view(-1, N)       # [num_graphs, N]


        # node_importance_func: [B, N]
        abs_vals = node_importance_func.abs()
        sum_vals = abs_vals.sum(dim=1, keepdim=True)
        node_importance_func_norm = abs_vals / (sum_vals + 1e-8)


        print('node_importance_func_norm:',node_importance_func_norm.shape)

        # === 结构模态 IG ===
        attr_struct, _ = ig.attribute(
            inputs=bs.x,
            baselines=baseline_struct,
            n_steps=10,
            target=pred_classes_nodes,
            additional_forward_args=(
                bf.x, bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bs.y, bs.batch, t1_batch, asl_batch, 'struct'
            ),
            return_convergence_delta=True
        )
        node_importance_struct = attr_struct.abs().mean(dim=1)
        node_importance_struct = node_importance_struct.view(-1, N)   # [num_graphs, N]
        # node_importance_func: [B, N]
        abs_vals = node_importance_struct.abs()
        sum_vals = abs_vals.sum(dim=1, keepdim=True)
        node_importance_struct_norm = abs_vals / (sum_vals + 1e-8)


        # === 平均融合 ===
        node_importance_mean = (node_importance_func_norm + node_importance_struct_norm) / 2
        abs_vals = node_importance_mean.abs()
        sum_vals = abs_vals.sum(dim=1, keepdim=True)
        node_importance_mean_norm = abs_vals / (sum_vals + 1e-8)

        all_func.append(node_importance_func_norm)
        all_struct.append(node_importance_struct_norm)
        all_mean.append(node_importance_mean_norm)

    # === 跨 batch 平均（受试者平均） ===
    node_importance_func_final = torch.cat(all_func, dim=0).mean(dim=0)
    node_importance_struct_final = torch.cat(all_struct, dim=0).mean(dim=0)
    node_importance_mean_final = torch.cat(all_mean, dim=0).mean(dim=0)


    return node_importance_func_final, node_importance_struct_final, node_importance_mean_final


def node_importance_joint(func_graphs, struct_graphs, model, loader, device):
    model.eval()
    all_joint = []
    all_joint_delta = []

    for batch_idx in loader:
        bf = Batch.from_data_list([func_graphs[i] for i in batch_idx]).to(device)
        bs = Batch.from_data_list([struct_graphs[i] for i in batch_idx]).to(device)

        # 节点级预测
        with torch.no_grad():
            logits = model.forward(
                bf.x, bf.edge_index, bf.edge_attr,
                bs.x, bs.edge_index, bs.edge_attr,
                bf.y, bf.batch, mode='eval'
            )
            pred_classes = logits.argmax(dim=1)  # [batch]

        ig = IntegratedGradients(model.forward_for_captum_joint)
        # baselines must be a tuple matching inputs
        baselines = (torch.zeros_like(bf.x), torch.zeros_like(bs.x))

        joint_attr, joint_delta = ig.attribute(
            inputs=(bf.x, bs.x),
            baselines=baselines,
            n_steps=10,
            return_convergence_delta=True,
            additional_forward_args=(
                bf.edge_index, bf.edge_attr,
                bs.edge_index, bs.edge_attr,
                bf.y, bf.batch
            )
        )
        attr_bf, attr_bs = joint_attr
        attr_bf = attr_bf.view(-1, 90, 2)[:, :, 1]  # 类别1
        attr_bs = attr_bs.view(-1, 90, 2)[:, :, 1]
        joint_attr = attr_bf + attr_bs  # 简单相加

        mask = (pred_classes == 1)
        joint_attr_masked = joint_attr[mask]

        all_joint.append(joint_attr_masked)
        all_joint_delta.append(joint_delta.abs())

    # 合并所有 batch
    all_joint = torch.cat(all_joint, dim=0)  # [num_selected_samples, num_nodes]
    node_importance_joint = all_joint.mean(dim=0)
    avg_joint_delta = torch.cat(all_joint_delta).mean().item()

    print('joint_delta', avg_joint_delta)

    return node_importance_joint,  avg_joint_delta


def normalize_attributions(attributions_list, num_nodes=90):
    """
    ✅ 修复版：对每一折的节点贡献度进行归一化并取平均

    主要改进：
    1. 先跨折平均（保留原始量级）
    2. 再统一归一化

    Args:
        attributions_list: list，每个元素是一折的结果 [num_nodes] 或 [B, num_nodes]
        num_nodes: 每个图的节点数（默认90）

    Returns:
        avg_attr_norm: [num_nodes] 的 tensor，归一化后的跨折平均结果
    """
    processed_list = []

    for attr in attributions_list:
        # 保证是 float tensor
        if not torch.is_tensor(attr):
            attr = torch.tensor(attr, dtype=torch.float32)
        else:
            attr = attr.float()

        # 若为多受试者拼接的情况，先求平均
        if attr.numel() > num_nodes:
            num_graphs = attr.numel() // num_nodes
            attr = attr.view(num_graphs, num_nodes).mean(dim=0)

        processed_list.append(attr)

    # ===== 修复：先跨折平均，再归一化 =====
    avg_attr = torch.stack(processed_list, dim=0).mean(dim=0)  # [90]

    # 统一 Min-Max 归一化
    avg_attr_norm = normalize_final(avg_attr)

    return avg_attr_norm
