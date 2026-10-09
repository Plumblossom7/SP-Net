import torch
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, APPNP, GATConv, GATv2Conv, global_max_pool
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_softmax, scatter_sum
import math
# ------------------ 损失函数 ------------------ #

def bidirectional_contrastive_loss(z_fc, z_struct, tau=0.07, lambda_param=0.5):
    """
    双向对比损失，同时考虑fc→struct和struct→fc两个方向
    """
    batch_size = z_fc.size(0)

    # 1. L2归一化
    z_fc_norm = F.normalize(z_fc, p=2, dim=1)
    z_struct_norm = F.normalize(z_struct, p=2, dim=1)

    # 2. 计算相似度矩阵并应用温度参数
    sim_fc_struct = torch.mm(z_fc_norm, z_struct_norm.t()) / tau  # [N, N]
    sim_struct_fc = sim_fc_struct.t()  # 转置获得另一方向

    # 3. 创建对角线掩码
    mask = torch.eye(batch_size, device=z_fc.device)

    # 4. 第一个方向：fc→struct
    # 正样本和负样本分离
    pos_sim_fc = torch.diagonal(sim_fc_struct)  # [N]
    exp_sim_fc = torch.exp(sim_fc_struct)
    neg_exp_sim_fc = exp_sim_fc * (1 - mask)  # 屏蔽对角线
    # 计算每个样本的损失
    per_sample_loss_fc = -torch.log(
        torch.exp(pos_sim_fc) / (torch.exp(pos_sim_fc) + neg_exp_sim_fc.sum(dim=1))
    )

    # 5. 第二个方向：struct→fc
    # 正样本和负样本分离
    pos_sim_struct = torch.diagonal(sim_struct_fc)  # [N]
    exp_sim_struct = torch.exp(sim_struct_fc)
    neg_exp_sim_struct = exp_sim_struct * (1 - mask)  # 屏蔽对角线
    # 计算每个样本的损失
    per_sample_loss_struct = -torch.log(
        torch.exp(pos_sim_struct) / (torch.exp(pos_sim_struct) + neg_exp_sim_struct.sum(dim=1))
    )

    # 6. 合并两个方向的损失
    loss_fc_struct = per_sample_loss_fc.mean()
    loss_struct_fc = per_sample_loss_struct.mean()

    # 7. 加权平均
    total_loss = lambda_param * loss_fc_struct + (1 - lambda_param) * loss_struct_fc

    return total_loss

def bidirectional_node_contrastive_loss(z_fc, z_struct, tau=0.07, lambda_param=0.5):
    """
    双向节点级对比损失（负样本只来自不同样本节点）
    z_fc, z_struct: [batch, num_nodes, feature]
    tau: 温度参数
    lambda_param: 双向加权
    """
    B, N, F_dim = z_fc.size()

    # 1. L2归一化
    z_fc_norm = F.normalize(z_fc, p=2, dim=2)      # [B, N, F]
    z_struct_norm = F.normalize(z_struct, p=2, dim=2)  # [B, N, F]

    # 2. 节点级相似度矩阵
    sim = torch.bmm(z_fc_norm, z_struct_norm.transpose(1,2)) / tau  # [B, N, N]

    # 3. 对角线为正样本
    pos_sim = torch.diagonal(sim, dim1=1, dim2=2)  # [B, N]

    # 4. 构建负样本 mask，只考虑不同样本节点
    # mask_same_sample: 1表示同一图节点，0表示不同图节点
    mask_same_sample = torch.eye(N, device=z_fc.device).unsqueeze(0).repeat(B,1,1)

    # 负样本仅来自不同样本节点
    # 展开 batch 维度，将所有节点放在一起
    z_fc_flat = z_fc_norm.view(B*N, F_dim)
    z_struct_flat = z_struct_norm.view(B*N, F_dim)

    # 相似度矩阵：[B*N, B*N]
    sim_flat = torch.mm(z_fc_flat, z_struct_flat.t()) / tau

    # 构建正样本 mask（同一个节点）: [B*N]
    pos_idx = torch.arange(B*N, device=z_fc.device)

    # 构建负样本 mask: 屏蔽同一个图内节点和自己
    batch_idx = torch.arange(B, device=z_fc.device).repeat_interleave(N)
    node_idx = torch.arange(N, device=z_fc.device).repeat(B)

    # mask: 1表示屏蔽，0表示负样本
    neg_mask = (batch_idx[:,None] == batch_idx[None,:]) & (node_idx[:,None] != node_idx[None,:])

    # 计算 loss
    exp_sim_flat = torch.exp(sim_flat)
    # 正样本
    pos_sim_flat = sim_flat[pos_idx, pos_idx]
    # 负样本
    neg_exp_flat = exp_sim_flat * (~neg_mask).float()  # 只保留不同样本节点
    neg_sum_flat = neg_exp_flat.sum(dim=1)

    loss_fc_struct = -torch.log(torch.exp(pos_sim_flat) / (torch.exp(pos_sim_flat) + neg_sum_flat))
    loss_fc_struct = loss_fc_struct.mean()

    # 第二个方向 struct -> fc
    sim_flat2 = sim_flat.t()
    pos_sim_flat2 = sim_flat2[pos_idx, pos_idx]
    neg_exp_flat2 = torch.exp(sim_flat2) * (~neg_mask).float()
    neg_sum_flat2 = neg_exp_flat2.sum(dim=1)
    loss_struct_fc = -torch.log(torch.exp(pos_sim_flat2) / (torch.exp(pos_sim_flat2) + neg_sum_flat2))
    loss_struct_fc = loss_struct_fc.mean()

    total_loss = lambda_param * loss_fc_struct + (1 - lambda_param) * loss_struct_fc
    return total_loss

def orthogonal_loss(z_shared, z_private):
    """计算正交损失，鼓励共享和私有(不同模态的私有）嵌入正交。"""
    # 添加 eps=1e-8 防止除以零
    s = F.normalize(z_shared, p=2, dim=1, eps=1e-8)
    p = F.normalize(z_private, p=2, dim=1, eps=1e-8)
    # (s * p).sum(dim=1) is the dot product (cosine similarity here)
    return (s * p).sum(dim=1).pow(2).mean()

def orthogonal_loss_node(z_shared, z_private, eps=1e-8):
    """
    节点级正交损失
    z_shared, z_private: [batch, num_nodes, feature]
    """
    # L2归一化
    s = F.normalize(z_shared, p=2, dim=2, eps=eps)  # [B, N, F]
    p = F.normalize(z_private, p=2, dim=2, eps=eps) # [B, N, F]

    # 计算每个节点的余弦相似度
    # (s * p).sum(dim=2) -> [B, N]
    cos_sim = (s * p).sum(dim=2)

    # 平方并平均
    loss = (cos_sim ** 2).mean()

    return loss

def correlation_penalty(x, y, eps=1e-8):
    """
    去相关损失（decorrelation loss）
    通过最小化协方差矩阵的Frobenius范数来降低模态间的线性相关性。
    输入:
        x, y: [batch_size, feature_dim]
    返回:
        loss: 标量，值越小说明相关性越低
    """
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    cov = (x.T @ y) / (x.size(0) - 1)
    loss = torch.norm(cov, p='fro')  # Frobenius范数
    return loss

# ------------------ 模型组件 ------------------ #

class GraphEncoder(nn.Module):
    """
    图编码器：两层GCN，将节点特征编码后通过全局平均池化得到图级表示。
    """
    def __init__(self, in_dim, hidden): #hidden_dim是每个节点的特征维度？
        super().__init__()
        self.hidden = hidden
        self.conv1 = GCNConv(in_dim, hidden//2)
        self.conv2 = GCNConv(hidden//2, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn_node = nn.BatchNorm1d(hidden)
        self.ln_node = nn.LayerNorm(hidden)
        # # 添加节点注意力层
        # self.node_attention = NodeAttention(hidden)


    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight).tanh()# (N, R, hidden//2)
        x = self.conv2(x, edge_index, edge_weight).tanh()
        x = self.ln_node(x)#(N, R, hidden)

        return x

class GraphEncoder_GraphPool(nn.Module):
    """
    图编码器：两层GCN，将节点特征编码后通过全局平均池化得到图级表示。
    """
    def __init__(self, in_dim, hidden): #hidden_dim是每个节点的特征维度？
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden//2)
        self.conv2 = GCNConv(hidden//2, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn_node = nn.BatchNorm1d(hidden)
        self.ln_node = nn.LayerNorm(hidden)
        # # 添加节点注意力层
        # self.node_attention = NodeAttention(hidden)


    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        #print("x.shape:", x.shape)
        edge_weight = data.edge_attr if data.edge_attr is not None else None
        x = self.conv1(x, edge_index, edge_weight).tanh()# (N, R, hidden//2)
        x = self.conv2(x, edge_index, edge_weight).tanh()
        #print("x.shape:", x.shape)
        #x = self.conv3(x, edge_index, edge_weight).tanh()
        x = self.ln_node(x)#(N, R, hidden)

        # # 重塑为[batch_size, nodes_per_graph, hidden]格式
        # batch_size = int(batch.max().item() + 1)
        # nodes_per_graph = 90  # 假设是AAL90脑图谱
        # x_reshaped = x.reshape(batch_size, nodes_per_graph, -1)

        # # 应用节点间注意力
        # x_attended, attn_weights = self.node_attention(x_reshaped)

        #print("x.shape:", x.shape)


        # 使用正确的方式进行池化
        # 方法1：直接在3D张量上进行池化，加了个read out
        mean_pool = global_mean_pool(x, batch)  # [batch_size, hidden]
        max_pool = global_max_pool(x, batch)  # [batch_size, hidden]
        #print("mean_pool.shape:", mean_pool.shape)
        #print("max_pool.shape:", max_pool.shape)


        # 拼接不同的池化结果
        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)  # [batch_size, 2*hidden]

        return graph_embedding

class GraphEncoder_NodePool(nn.Module):
    """
    图编码器：两层GCN，将节点特征编码后通过全局平均池化得到图级表示。
    """
    def __init__(self, in_dim, hidden): #hidden_dim是每个节点的特征维度？
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden//2)
        self.conv2 = GCNConv(hidden//2, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn_node = nn.BatchNorm1d(hidden)
        self.ln_node = nn.LayerNorm(hidden)
        # # 添加节点注意力层
        # self.node_attention = NodeAttention(hidden)


    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        #print("x.shape:", x.shape)
        edge_weight = data.edge_attr if data.edge_attr is not None else None
        x = self.conv1(x, edge_index, edge_weight).tanh()# (N, R, hidden//2)
        x = self.conv2(x, edge_index, edge_weight).tanh()
        #print("x.shape:", x.shape)
        #x = self.conv3(x, edge_index, edge_weight).tanh()
        x = self.ln_node(x)#(N, R, hidden)

        # 重塑为[batch_size, nodes_per_graph, hidden]格式
        batch_size = int(batch.max().item() + 1)
        nodes_per_graph = 90  # 假设是AAL90脑图谱
        x_reshaped = x.reshape(batch_size, nodes_per_graph, -1)

        # 假设 x 是 [batch_size, num_nodes, hidden]
        x_node_mean = x.mean(dim=-1, keepdim=True)  # [batch_size, num_nodes, 1]
        x_node_max = x.max(dim=-1, keepdim=True).values  # [batch_size, num_nodes, 1]
        x_node_pooled = torch.cat([x_node_mean, x_node_max], dim=-1)  # [batch_size, num_nodes, 2]

        return x_node_pooled

class CrossModalAttention_Bidirection(nn.Module):
    """跨模态注意力层：让两个模态特征相互增强"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x1, x2):
        # x1作为查询，x2作为键和值
        q1 = self.q_proj(x1)
        k2 = self.k_proj(x2)
        v2 = self.v_proj(x2)

        # 计算注意力权重
        attn = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 注意力加权
        x1_enhanced = torch.matmul(attn, v2)

        # 同样计算x2增强by x1
        q2 = self.q_proj(x2)
        k1 = self.k_proj(x1)
        v1 = self.v_proj(x1)
        attn = torch.matmul(q2, k1.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x2_enhanced = torch.matmul(attn, v1)

        # 残差连接
        x1_out = x1 + x1_enhanced
        x2_out = x2 + x2_enhanced

        return x1_out, x2_out

class CrossModalAttention_Unidirection(nn.Module):
    """共享特征作为查询，原始特征作为键和值，使得原始特征增强，不一定是本模态的共享特征"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x1, x2):
        # x1作为查询，x2作为键和值
        q1 = self.q_proj(x1)
        k2 = self.k_proj(x2)
        v2 = self.v_proj(x2)

        # 计算注意力权重
        attn = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 注意力加权
        x2_enhanced = torch.matmul(attn, v2)

        # 残差连接
        x2_out = x2 + x2_enhanced

        return x2_out

class CrossModalAttention_Mix(nn.Module):
    """共享特征作为查询，原始特征作为键和值，使得原始特征增强，不一定是本模态的共享特征"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x1, x2, x3):
        # x1作为查询，x2作为键和值
        q1 = self.q_proj(x1)
        k2 = self.k_proj(x2)
        v3 = self.v_proj(x3)

        # 计算注意力权重
        attn = torch.matmul(q1, k2.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 注意力加权
        x3_enhanced = torch.matmul(attn, v3)

        # 残差连接
        x3_out = x3 + x3_enhanced

        return x3_out

class SelfAttention(nn.Module):
    """跨模态注意力层：让两个模态特征相互增强"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        # x1作为查询，x2作为键和值
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 计算注意力权重
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 注意力加权
        x_enhanced = torch.matmul(attn, v)

        # 残差连接
        x_out = x + x_enhanced

        return x_out


class SharedPrivateHead(nn.Module):
    """
    将图级表示映射到共享和私有空间。
    """
    def __init__(self, in_dim, shared_dim, private_dim):
        super().__init__()
        # 这里可以使用MLP或者未来替换为KAN
        self.shared_mlp = nn.Linear(in_dim, shared_dim)
        self.private_mlp = nn.Linear(in_dim, private_dim)

    def forward(self, x):
        zs = self.shared_mlp(x)
        zp = self.private_mlp(x)
        return zs, zp


class SharedPrivateHead_pool(nn.Module):
    """
    将图级表示映射到共享和私有空间。
    """
    def __init__(self, in_dim, shared_dim, private_dim):
        super().__init__()

    def forward(self, x):
        # 假设 x 是 [batch_size, num_nodes, hidden]
        zs = x.mean(dim=-1, keepdim=True)  # [batch_size, num_nodes, 1]
        zp = x.max(dim=-1, keepdim=True).values  # [batch_size, num_nodes, 1]
        return zs, zp

class SharedPrivateHead_BN(nn.Module):
    def __init__(self, in_dim, shared_dim, private_dim):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )
        self.private_mlp = nn.Sequential(
            nn.Linear(in_dim, private_dim),
            nn.BatchNorm1d(private_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.shared_mlp(x), self.private_mlp(x)


class SharedPrivateHead_LN(nn.Module):
    def __init__(self, in_dim, shared_dim, private_dim):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU()
        )
        self.private_mlp = nn.Sequential(
            nn.Linear(in_dim, private_dim),
            nn.LayerNorm(private_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.shared_mlp(x), self.private_mlp(x)


class ClusterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):
        """
        embeddings: [batch_size, hidden_dim]
        labels: [batch_size]  每个样本的类别
        """
        device = embeddings.device
        unique_classes = labels.unique()

        intra_class_var = 0.0   # 类内方差之和
        inter_class_dist = 0.0  # 类间中心距离之和

        class_means = {}

        # 先计算每个类别的中心
        for c in unique_classes:
            mask = (labels == c)
            emb_c = embeddings[mask]  # 该类所有 embedding
            mu_c = emb_c.mean(dim=0)
            class_means[c.item()] = mu_c
            # 类内方差（平方差的均值）
            intra_class_var += ((emb_c - mu_c) ** 2).mean()

        # 计算类间距离
        class_ids = list(class_means.keys())
        for i in range(len(class_ids)):
            for j in range(i+1, len(class_ids)):
                mu_i, mu_j = class_means[class_ids[i]], class_means[class_ids[j]]
                inter_class_dist += torch.norm(mu_i - mu_j, p=2) ** 2

        # 公式： log ( exp(Σ σ²) / exp(Σ ||μ_c - μ_i||² ) )
        # 等价于 (Σ σ²) - (Σ ||μ_c - μ_i||²)
        loss = intra_class_var - inter_class_dist
        return loss

import torch
import torch.nn as nn

class ClusterLoss_new(nn.Module):
    def __init__(self, alpha=1.0):
        """
        alpha: 类间距离权重系数，控制类内 vs 类间的相对重要性
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, embeddings, labels):
        """
        embeddings: [batch_size, hidden_dim]
        labels: [batch_size]  每个样本的类别
        """
        device = embeddings.device
        unique_classes = labels.unique()
        num_classes = len(unique_classes)

        intra_class_var = 0.0   # 类内方差
        inter_class_dist = 0.0  # 类间中心距离

        class_means = {}

        # 计算每个类别的中心和类内方差
        for c in unique_classes:
            mask = (labels == c)
            emb_c = embeddings[mask]  # 该类所有 embedding
            mu_c = emb_c.mean(dim=0)
            class_means[c.item()] = mu_c
            # 类内方差加权，避免大类主导
            weight = 1.0 / mask.sum().item()
            intra_class_var += weight * ((emb_c - mu_c) ** 2).mean()

        # 计算类间距离，归一化
        class_ids = list(class_means.keys())
        count = 0
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                mu_i, mu_j = class_means[class_ids[i]], class_means[class_ids[j]]
                inter_class_dist += torch.norm(mu_i - mu_j, p=2) ** 2
                count += 1
        if count > 0:
            inter_class_dist /= count  # 类间距离归一化

        # 最终 loss: 类内方差 - alpha * 类间距离
        loss = intra_class_var - self.alpha * inter_class_dist
        return loss


class Fusion_P_S_E(nn.Module):

    def __init__(self, hidden_dim, shared_dim, private_dim):
        super(Fusion_P_S_E, self).__init__()
        self.cross_attn_e = CrossModalAttention_Bidirection(hidden_dim)
        self.cross_attn_s = CrossModalAttention_Bidirection(shared_dim)
        self.cross_attn_p = CrossModalAttention_Bidirection(private_dim)

    def forward(self, emb_1, emb_2, zs_1, zs_2, zp_1, zp_2):
        emb_12, emb_21 = self.cross_attn_e(emb_1, emb_2)
        zs_12, zs_21 = self.cross_attn_s(zs_1, zs_2)
        zp_12, zp_21 = self.cross_attn_p(zp_1, zp_2)
        fusion = torch.cat([emb_12, emb_21, zs_12, zs_21, zp_12, zp_21], dim=1)
        return fusion

class StableGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.1):
        super().__init__()
        self.hidden = hidden

        # 三层GCN，每层后都有归一化和激活
        self.conv1 = GCNConv(in_dim, hidden//2)
        self.bn1 = nn.BatchNorm1d(hidden//2)
        self.ln1 = nn.LayerNorm(hidden//2)

        self.conv2 = GCNConv(hidden//2, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.ln2 = nn.LayerNorm(hidden)

        self.conv3 = GCNConv(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.ln3 = nn.LayerNorm(hidden)

        # 残差连接
        self.residual = nn.Linear(in_dim, hidden) if in_dim != hidden else nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight, batch):
        # 保存原始输入用于残差连接
        identity = x

        # 第一层: 卷积 -> 归一化 -> 激活 -> dropout
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        # 第二层
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        # x = F.leaky_relu(x, negative_slope=0.1)
        # x = self.dropout(x)

        # # 第三层 + 残差连接
        # x = self.conv3(x, edge_index, edge_weight)
        # x = self.bn3(x)

        # 残差连接
        if isinstance(self.residual, nn.Linear):
            identity = self.residual(identity)
        x = x + identity  # 残差连接

        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        return x

class CrossSpecAttention(nn.Module):
    def __init__(self, d_model, d_k=None, dropout=0.1):
        super().__init__()
        d_k = d_k or d_model
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_k ** 0.5
        self.norm = nn.LayerNorm(d_model)

        # ✨ 新增：门控层
        self.gate_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, Q, K, V, mask=None):
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)      # [Qn, D]
        out = self.out(out)

        # ✨ 门控融合替代直接残差
        if out.shape == Q.shape:
            gate = torch.sigmoid(self.gate_proj(torch.cat([out, Q], dim=-1)))
            out = gate * out + (1 - gate) * Q
            out = self.norm(out)

        return out, attn



#节点级表示而不是图级表示
class MultiModalGNN(nn.Module):
    """
    多模态图神经网络模型，集成了图编码、共享/私有嵌入分离和损失计算。

    流程:
    1. 三种模态数据 (fc, gm, fa) 分别通过独立的 GraphEncoder 得到图级表示。
    2. 每个图级表示通过 SharedPrivateHead 分解为共享嵌入和私有嵌入。
    3. 计算损失或进行分类:
        - 训练时，计算对比损失和正交损失。
        - 推理时，融合共享嵌入并进行分类。
    """
    def __init__(self, in_dim1, in_dim2, shared_dim, private_dim, num_classes, tau=0.07, lambda_param=0.5, hidden=4, learnable_loss_weights=False, norm='None'):
        super().__init__()
        if learnable_loss_weights:
            self.lcont = nn.Parameter(torch.tensor(1.0))
            self.lothro = nn.Parameter(torch.tensor(1.0))
        else:
            self.lcont = None
            self.lothro = None

        self.tau = tau
        self.lambda_param = lambda_param

        # 用来缓存固定的图结构
        self.edge_index_bf = None
        self.edge_weight_bf = None
        self.edge_index_bs = None
        self.edge_weight_bs = None
        self.batch = None

        #先用一个1*1卷积给拼接一起的不同特征分配一下权重
        self.conv_func = nn.Conv1d(in_channels=in_dim1, out_channels=in_dim1, kernel_size=1)
        self.conv_struct = nn.Conv1d(in_channels=in_dim2, out_channels=in_dim2, kernel_size=1)

        # 为每个模态创建一个图编码器
        # self.encoder_func = GATWithAttentionPooling(in_dim1, hidden, hidden)  # 功能模态
        # self.encoder_struct = GATWithAttentionPooling(in_dim2, hidden, hidden)    # 结构模态
        #两层
        # self.encoder_func = GraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        # self.encoder_struct = GraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)

        self.encoder_func = StableGraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        self.encoder_struct = StableGraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)
        # 输出维度
        self.hidden_dim = hidden



        # 创建共享/私有头
        if norm == 'None':
            self.head_func = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            # self.head_func = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)
            # self.head_struct = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)

        elif norm == 'ln':
            self.head_func = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
        elif norm == 'bn':
            self.head_func = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)

        # 跨模态注意力层
        self.cross_attn_func_struct = CrossModalAttention_Bidirection(self.hidden_dim)

        #分层次跨模态融合
        #self.cross_attn_P_S_E = Fusion_P_S_E(self.hidden_dim, shared_dim, private_dim)

        # 将共享特征映射到与嵌入相同的维度，便于融合
        self.linear = nn.Linear(shared_dim, self.hidden_dim)# 将共享特征映射到与嵌入相同的维度
        # 让共享特征增强原始特征
        self.cross_attn_S_E = CrossModalAttention_Unidirection(self.hidden_dim)

        self.cluster_loss = ClusterLoss()

        # 分类器：输入为共享+私有2模态拼接
        #total_dim = shared_dim * 2 + private_dim * 2  # 1个共享 + 2个私有

        #self.conv1x1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.conv1x1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)


        #total_dim = shared_dim
        #total_dim = 2*self.hidden_dim
        total_dim = 2*self.hidden_dim + 2*shared_dim + 2*private_dim
        #total_dim = 4*(self.hidden_dim + 2)
        # 使用更复杂的分类器 - MLP with BatchNorm and Dropout
        self.classifier_bn = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.BatchNorm1d(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.BatchNorm1d(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_dim // 2, num_classes)
        )
        self.classifier_ln = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.LayerNorm(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(total_dim // 2, total_dim // 4),
            nn.LayerNorm(total_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(total_dim // 4, num_classes)
        )

    def set_graph(self, bf_edge_index, bf_edge_weight,
                  bs_edge_index, bs_edge_weight, batch):
        """存储固定的图结构，供归因时使用"""
        self.edge_index_bf = bf_edge_index
        self.edge_weight_bf = bf_edge_weight
        self.edge_index_bs = bs_edge_index
        self.edge_weight_bs = bs_edge_weight
        self.batch = batch

    def mean_max_pool(self, x):
        mean_pool = x.mean(dim=1)   # [batch, feature]
        max_pool, _ = x.max(dim=1)  # [batch, feature]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2*feature]

    def forward_for_captum(self, inputs, other_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch, mode='func'):

        if mode == 'func':
            bf_x = inputs
            bs_x = other_x

        elif mode == 'struct':
            bs_x = inputs
            bf_x = other_x

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_for_captum_joint(self, bf_x, bs_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch):

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_return_embeddings(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch):
        """
        只返回中间节点嵌入（共享/私有）。
        不做池化、分类或损失计算。
        """
        # # 1. 1*1卷积加权
        x_func = bf_x.view(-1, 90, bf_x.shape[1]).permute(0,2,1)
        x_func = self.conv_func(x_func)
        bf_x = x_func.permute(0,2,1).reshape(-1, bf_x.shape[1])

        x_struct = bs_x.view(-1, 90, bs_x.shape[1]).permute(0,2,1)
        x_struct = self.conv_struct(x_struct)
        bs_x = x_struct.permute(0,2,1).reshape(-1, bs_x.shape[1])

        # 2. 节点级编码
        emb_func = self.encoder_func(bf_x, bf_edge_index, bf_edge_weight, batch)   # [batch, nodes, hidden]
        emb_struct = self.encoder_struct(bs_x, bs_edge_index, bs_edge_weight, batch)

        # print("emb_func.shape:", emb_func.shape)
        # print("emb_struct.shape:", emb_struct.shape)

        # 3. 分解共享/私有嵌入
        zs_func, zp_func = self.head_func(emb_func)
        zs_struct, zp_struct = self.head_struct(emb_struct)

        # 4. 可选跨模态注意力增强

        emb_func, emb_struct = self.cross_attn_func_struct(emb_func, emb_struct)

        return {
            'emb_func': emb_func,
            'emb_struct': emb_struct,
            'zs_func': zs_func,
            'zs_struct': zs_struct,
            'zp_func': zp_func,
            'zp_struct': zp_struct
        }

    def forward(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, labels, batch, mode='train'):


        # 获取中间嵌入
        embeddings = self.forward_return_embeddings(bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch)

        emb_func = embeddings['emb_func']
        emb_struct = embeddings['emb_struct']
        zs_func = embeddings['zs_func']
        zs_struct = embeddings['zs_struct']
        zp_func = embeddings['zp_func']
        zp_struct = embeddings['zp_struct']

        # --- 训练/测试逻辑 ---
        if mode == 'train':
            # 对比损失、正交损失
            contrastive_loss = bidirectional_contrastive_loss(zs_func, zs_struct, self.tau, self.lambda_param)
            ortho_loss = (orthogonal_loss(zs_func, zp_func)
                          + orthogonal_loss(zs_struct, zp_struct)
                          + orthogonal_loss(zp_func, zp_struct)) / 3

            # 聚类损失,使得共享特征真的具有与疾病有关的信息

            # cluster_loss_f = self.cluster_loss(zs_func, labels)
            # cluster_loss_s = self.cluster_loss(zs_struct, labels)
            # cluster_loss = (cluster_loss_f + cluster_loss_s) / 2


            # 图级特征拼接、池化
            final_embedding = torch.cat([emb_func, emb_struct, zs_func, zs_struct, zp_func, zp_struct], dim=-1)
            #print("final_embedding.shape before pool:", final_embedding.shape)
            batch_size = int(batch.max().item() + 1)
            final_embedding = final_embedding.reshape(batch_size, 90, -1)  # (B, N, F)
            final_embedding = final_embedding.mean(dim=1)  # (batch_size, hidden_dim)



            # 分类
            logits = self.classifier_bn(final_embedding)
            cluster_loss = 0
            return {
                'logits': logits,
                'contrastive_loss': contrastive_loss,
                'orthogonal_loss': ortho_loss,
                #'cluster_loss': cluster_loss,
            }
        elif mode == 'eval':
            # eval模式只做分类
            final_embedding = torch.cat([emb_func, emb_struct, zs_func, zs_struct, zp_func, zp_struct], dim=-1)
            batch_size = int(batch.max().item() + 1)
            final_embedding = final_embedding.reshape(batch_size, 90, -1)  # (B, N, F)
            final_embedding = final_embedding.mean(dim=1)  # (batch_size, hidden_dim)
            logits = self.classifier_bn(final_embedding)
            return logits
        elif mode == 'captum':
            final_embedding = torch.cat([emb_func, emb_struct, zs_func, zs_struct, zp_func, zp_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits


class MultiModalGNN_new(nn.Module):
    """
    多模态图神经网络模型，集成了图编码、共享/私有嵌入分离和损失计算。

    流程:
    1. 三种模态数据 (fc, gm, fa) 分别通过独立的 GraphEncoder 得到图级表示。
    2. 每个图级表示通过 SharedPrivateHead 分解为共享嵌入和私有嵌入。
    3. 计算损失或进行分类:
        - 训练时，计算对比损失和正交损失。
        - 推理时，融合共享嵌入并进行分类。
    """
    def __init__(self, in_dim1, in_dim2, shared_dim, private_dim, num_classes, tau=0.07, lambda_param=0.5, hidden=4, learnable_loss_weights=False, norm='None'):
        super().__init__()
        if learnable_loss_weights:
            self.lcont = nn.Parameter(torch.tensor(1.0))
            self.lothro = nn.Parameter(torch.tensor(1.0))
        else:
            self.lcont = None
            self.lothro = None

        self.tau = tau
        self.lambda_param = lambda_param
        self.alpha = 0.5

        # 用来缓存固定的图结构
        self.edge_index_bf = None
        self.edge_weight_bf = None
        self.edge_index_bs = None
        self.edge_weight_bs = None
        self.batch = None

        #先用一个1*1卷积给拼接一起的不同特征分配一下权重
        self.conv_func = nn.Conv1d(in_channels=in_dim1, out_channels=in_dim1, kernel_size=1)
        self.conv_struct = nn.Conv1d(in_channels=in_dim2, out_channels=in_dim2, kernel_size=1)

        # 为每个模态创建一个图编码器
        # self.encoder_func = GATWithAttentionPooling(in_dim1, hidden, hidden)  # 功能模态
        # self.encoder_struct = GATWithAttentionPooling(in_dim2, hidden, hidden)    # 结构模态
        #两层
        # self.encoder_func = GraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        # self.encoder_struct = GraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)

        self.encoder_func = StableGraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        self.encoder_struct = StableGraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)
        # 输出维度
        self.hidden_dim = hidden



        # 创建共享/私有头
        if norm == 'None':
            self.head_func = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            # self.head_func = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)
            # self.head_struct = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)

        elif norm == 'ln':
            self.head_func = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
        elif norm == 'bn':
            self.head_func = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)

        # 跨模态注意力层
        #self.cross_attn_func_struct = CrossModalAttention_Bidirection(self.hidden_dim)

        #分层次跨模态融合
        #self.cross_attn_P_S_E = Fusion_P_S_E(self.hidden_dim, shared_dim, private_dim)

        # 将共享特征映射到与嵌入相同的维度，便于融合
        self.linear = nn.Linear(self.hidden_dim, shared_dim)# 将嵌入特征映射到与共享相同的维度
        # 让共享特征增强原始特征
        self.cross_attn_S_E = CrossModalAttention_Mix(shared_dim)
        self.enhancer = CrossSpecAttention(d_model=shared_dim)

       # self.cluster_loss = ClusterLoss()

        # 分类器：输入为共享+私有2模态拼接
        #total_dim = shared_dim * 2 + private_dim * 2  # 1个共享 + 2个私有

        #self.conv1x1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.conv1x1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)


        #total_dim = shared_dim
        #total_dim = 2*self.hidden_dim
        total_dim = 2*shared_dim
        #total_dim = 4*(self.hidden_dim + 2)
        # 使用更复杂的分类器 - MLP with BatchNorm and Dropout
        self.classifier_bn = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.BatchNorm1d(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.BatchNorm1d(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_dim // 2, num_classes)
        )
        spe_dim = 2*shared_dim
        self.classifier_spe = nn.Sequential(
            nn.Linear(spe_dim, spe_dim // 2),
            nn.BatchNorm1d(spe_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.LayerNorm(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(spe_dim // 2, num_classes)
        )

    def set_graph(self, bf_edge_index, bf_edge_weight,
                  bs_edge_index, bs_edge_weight, batch):
        """存储固定的图结构，供归因时使用"""
        self.edge_index_bf = bf_edge_index
        self.edge_weight_bf = bf_edge_weight
        self.edge_index_bs = bs_edge_index
        self.edge_weight_bs = bs_edge_weight
        self.batch = batch

    def mean_max_pool(self, x):
        mean_pool = x.mean(dim=1)   # [batch, feature]
        max_pool, _ = x.max(dim=1)  # [batch, feature]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2*feature]

    def forward_for_captum(self, inputs, other_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch, mode='func'):

        if mode == 'func':
            bf_x = inputs
            bs_x = other_x

        elif mode == 'struct':
            bs_x = inputs
            bf_x = other_x

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_for_captum_joint(self, bf_x, bs_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch):

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_return_embeddings(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch):
        """
        只返回中间节点嵌入（共享/私有）。
        不做池化、分类或损失计算。
        """
        # # # 1. 1*1卷积加权
        # x_func = bf_x.view(-1, 90, bf_x.shape[1]).permute(0,2,1)
        # x_func = self.conv_func(x_func)
        # bf_x = x_func.permute(0,2,1).reshape(-1, bf_x.shape[1])

        # x_struct = bs_x.view(-1, 90, bs_x.shape[1]).permute(0,2,1)
        # x_struct = self.conv_struct(x_struct)
        # bs_x = x_struct.permute(0,2,1).reshape(-1, bs_x.shape[1])

        # 2. 节点级编码
        emb_func = self.encoder_func(bf_x, bf_edge_index, bf_edge_weight, batch)   # [batch, nodes, hidden]
        emb_struct = self.encoder_struct(bs_x, bs_edge_index, bs_edge_weight, batch)

        # print("emb_func.shape:", emb_func.shape)
        # print("emb_struct.shape:", emb_struct.shape)

        # 3. 分解共享/私有嵌入
        zs_func, zp_func = self.head_func(emb_func)
        zs_struct, zp_struct = self.head_struct(emb_struct)

        ## 4. 可选跨模态注意力增强

        #emb_func, emb_struct = self.cross_attn_func_struct(emb_func, emb_struct)

        return {
            'emb_func': emb_func,
            'emb_struct': emb_struct,
            'zs_func': zs_func,
            'zs_struct': zs_struct,
            'zp_func': zp_func,
            'zp_struct': zp_struct
        }

    def forward(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, labels, batch, mode='train'):


        # 获取中间嵌入
        embeddings = self.forward_return_embeddings(bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch)

        emb_func = embeddings['emb_func']
        emb_struct = embeddings['emb_struct']
        zs_func = embeddings['zs_func']
        zs_struct = embeddings['zs_struct']
        zp_func = embeddings['zp_func']
        zp_struct = embeddings['zp_struct']

        emb_func = self.linear(emb_func)
        emb_struct = self.linear(emb_struct)


        # --- 训练/测试逻辑 ---
        if mode == 'train':
            # 对比损失、正交损失
            contrastive_loss = bidirectional_contrastive_loss(zs_func, zs_struct, self.tau, self.lambda_param)
            ortho_loss = (orthogonal_loss(zs_func, zp_func)
                          + orthogonal_loss(zs_struct, zp_struct)
                          + orthogonal_loss(zp_func, zp_struct)) / 3

            #共享增强原始嵌入
            # stren_func = self.cross_attn_S_E(zs_func, emb_func) #zs_func是query
            # stren_struct = self.cross_attn_S_E(zs_struct, emb_struct) #zs_func是query
            # stren_func = self.cross_attn_S_E(zp_func, emb_func) #zs_func是query
            # stren_struct = self.cross_attn_S_E(zp_struct, emb_struct) #zs_func是query
            # 假设 zp_func, zs_func, emb_func 都是 [num_nodes_total, D]

            enhanced_func = zs_func + self.alpha * zp_func
            enhanced_struct = zs_struct + self.alpha * zp_struct


            # # 功能模态：zp_func 问，zs_func 当 Key，emb_func 当 Value
            # enhanced_func, attn_fs = self.enhancer(zp_func, zs_func, emb_func)

            # # 结构模态：zp_struct 问，zs_struct 当 Key，emb_struct 当 Value
            # enhanced_struct, attn_sf = self.enhancer(zp_struct, zs_struct, emb_struct)
            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)

            # # 结合到你的分类管道（例如 reshape -> mean pool）
            # # 若仍用节点级池化：先 reshape (B,N,D)
            # final_func = enhanced_func.reshape(batch_size, N, -1).mean(dim=1)
            # final_struct = enhanced_struct.reshape(batch_size, N, -1).mean(dim=1)
            # logits = classifier(torch.cat([final_func, final_struct], dim=-1))


            # 聚类损失,使得共享特征真的具有与疾病有关的信息

            # cluster_loss_f = self.cluster_loss(zs_func, labels)
            # cluster_loss_s = self.cluster_loss(zs_struct, labels)
            # cluster_loss = (cluster_loss_f + cluster_loss_s) / 2


            # 图级特征拼接、池化
            #final_embedding = torch.cat([emb_func, emb_struct, zs_func, zs_struct, zp_func, zp_struct], dim=-1)
            final_embedding = torch.cat([enhanced_func, enhanced_struct], dim=-1)
            # final_embedding = torch.cat([emb_func, emb_struct], dim=-1)
            #final_embedding = torch.cat([fused_func, fused_struct], dim=-1)
            #spe_embedding = torch.cat([zp_func, zp_struct], dim=-1)
            #print("final_embedding.shape before pool:", final_embedding.shape)
            batch_size = int(batch.max().item() + 1)
            final_embedding = final_embedding.reshape(batch_size, 90, -1)  # (B, N, F)
            final_embedding = final_embedding.mean(dim=1)  # (batch_size, hidden_dim)
            #spe_embedding = spe_embedding.reshape(batch_size, 90, -1)  # (B, N, F)
            #spe_embedding = spe_embedding.mean(dim=1)  # (batch_size, hidden_dim)

            # 分类
            logits = self.classifier_bn(final_embedding)
            #logits_spe = self.classifier_spe(spe_embedding)
            #cluster_loss = 0
            return {
                'logits': logits,
                #'logits_spe': logits_spe,
                'contrastive_loss': contrastive_loss,
                'orthogonal_loss': ortho_loss,
                #'cluster_loss': cluster_loss,
            }

        elif mode == 'eval':
            # eval模式只做分类
            #共享增强原始嵌入
            # stren_func = self.cross_attn_S_E(zp_func, emb_func) #zs_func是query
            # stren_struct = self.cross_attn_S_E(zp_struct, emb_struct) #zs_func是query
            # 功能模态：zp_func 问，zs_func 当 Key，emb_func 当 Value
            # enhanced_func, attn_fs = self.enhancer(emb_struct, emb_func, emb_func)

            # # 结构模态：zp_struct 问，zs_struct 当 Key，emb_struct 当 Value
            # enhanced_struct, attn_sf = self.enhancer(emb_func, emb_struct, emb_struct)
            enhanced_func = zs_func + self.alpha * zp_func
            enhanced_struct = zs_struct + self.alpha * zp_struct



            # # 功能模态：zp_func 问，zs_func 当 Key，emb_func 当 Value
            # enhanced_func, attn_fs = self.enhancer(zp_func, zs_func, emb_func)

            # # 结构模态：zp_struct 问，zs_struct 当 Key，emb_struct 当 Value
            # enhanced_struct, attn_sf = self.enhancer(zp_struct, zs_struct, emb_struct)
            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)
            #final_embedding = torch.cat([emb_func, emb_struct, zs_func, zs_struct, zp_func, zp_struct], dim=-1)
            #final_embedding = torch.cat([enhanced_func, enhanced_struct, zp_func, zp_struct, zs_func, zs_struct], dim=-1)
            final_embedding = torch.cat([fused_func, fused_struct], dim=-1)
            # final_embedding = torch.cat([emb_func, emb_struct], dim=-1)
            batch_size = int(batch.max().item() + 1)
            final_embedding = final_embedding.reshape(batch_size, 90, -1)  # (B, N, F)
            final_embedding = final_embedding.mean(dim=1)  # (batch_size, hidden_dim)
            logits = self.classifier_bn(final_embedding)
            return logits

        elif mode == 'captum':
            #共享增强原始嵌入
            # stren_func = self.cross_attn_S_E(zp_func, emb_func) #zs_func是query
            # stren_struct = self.cross_attn_S_E(zp_struct, emb_struct) #zs_func是query
            # 功能模态：zp_func 问，zs_func 当 Key，emb_func 当 Value
            enhanced_func, attn_fs = self.enhancer(zp_func, zs_func, emb_func)

            # 结构模态：zp_struct 问，zs_struct 当 Key，emb_struct 当 Value
            enhanced_struct, attn_sf = self.enhancer(zp_struct, zs_struct, emb_struct)
            final_embedding = torch.cat([enhanced_func, enhanced_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits


class MultiModalGNN_new_new(nn.Module):
    """
    多模态图神经网络模型，集成了图编码、共享/私有嵌入分离和损失计算。

    流程:
    1. 三种模态数据 (fc, gm, fa) 分别通过独立的 GraphEncoder 得到图级表示。
    2. 每个图级表示通过 SharedPrivateHead 分解为共享嵌入和私有嵌入。
    3. 计算损失或进行分类:
        - 训练时，计算对比损失和正交损失。
        - 推理时，融合共享嵌入并进行分类。
    """
    def __init__(self, in_dim1, in_dim2, shared_dim, private_dim, num_classes, tau=0.07, lambda_param=0.5, hidden=4, learnable_loss_weights=False, norm='None'):
        super().__init__()
        if learnable_loss_weights:
            self.lcont = nn.Parameter(torch.tensor(1.0))
            self.lothro = nn.Parameter(torch.tensor(1.0))
        else:
            self.lcont = None
            self.lothro = None

        self.tau = tau
        self.lambda_param = lambda_param
        self.alpha = 0.5

        # 用来缓存固定的图结构
        self.edge_index_bf = None
        self.edge_weight_bf = None
        self.edge_index_bs = None
        self.edge_weight_bs = None
        self.batch = None

        #先用一个1*1卷积给拼接一起的不同特征分配一下权重
        self.conv_func = nn.Conv1d(in_channels=in_dim1, out_channels=in_dim1, kernel_size=1)
        self.conv_struct = nn.Conv1d(in_channels=in_dim2, out_channels=in_dim2, kernel_size=1)

        # 为每个模态创建一个图编码器
        # self.encoder_func = GATWithAttentionPooling(in_dim1, hidden, hidden)  # 功能模态
        # self.encoder_struct = GATWithAttentionPooling(in_dim2, hidden, hidden)    # 结构模态
        #两层
        # self.encoder_func = GraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        # self.encoder_struct = GraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)

        self.encoder_func = StableGraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        self.encoder_struct = StableGraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)
        # 输出维度
        self.hidden_dim = hidden



        # 创建共享/私有头
        if norm == 'None':
            self.head_func = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            # self.head_func = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)
            # self.head_struct = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)

        elif norm == 'ln':
            self.head_func = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
        elif norm == 'bn':
            self.head_func = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)

        # 跨模态注意力层
        #self.cross_attn_func_struct = CrossModalAttention_Bidirection(self.hidden_dim)

        #分层次跨模态融合
        #self.cross_attn_P_S_E = Fusion_P_S_E(self.hidden_dim, shared_dim, private_dim)

        # 将共享特征映射到与嵌入相同的维度，便于融合
        self.linear = nn.Linear(self.hidden_dim, shared_dim)# 将嵌入特征映射到与共享相同的维度
        # 让共享特征增强原始特征
        self.cross_attn_S_E = CrossModalAttention_Mix(shared_dim)
        self.enhancer = CrossSpecAttention(d_model=shared_dim)

       # self.cluster_loss = ClusterLoss()

        # 分类器：输入为共享+私有2模态拼接
        #total_dim = shared_dim * 2 + private_dim * 2  # 1个共享 + 2个私有

        #self.conv1x1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.conv1x1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)


        #total_dim = shared_dim
        #total_dim = 2*self.hidden_dim
        total_dim = 2*shared_dim
        #total_dim = 4*(self.hidden_dim + 2)
        # 使用更复杂的分类器 - MLP with BatchNorm and Dropout
        self.classifier_bn = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.BatchNorm1d(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.BatchNorm1d(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_dim // 2, num_classes)
        )
        spe_dim = 2*shared_dim
        self.classifier_spe = nn.Sequential(
            nn.Linear(spe_dim, spe_dim // 2),
            nn.BatchNorm1d(spe_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.LayerNorm(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(spe_dim // 2, num_classes)
        )

    def set_graph(self, bf_edge_index, bf_edge_weight,
                  bs_edge_index, bs_edge_weight, batch):
        """存储固定的图结构，供归因时使用"""
        self.edge_index_bf = bf_edge_index
        self.edge_weight_bf = bf_edge_weight
        self.edge_index_bs = bs_edge_index
        self.edge_weight_bs = bs_edge_weight
        self.batch = batch

    def mean_max_pool(self, x):
        mean_pool = x.mean(dim=1)   # [batch, feature]
        max_pool, _ = x.max(dim=1)  # [batch, feature]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2*feature]

    def forward_for_captum(self, inputs, other_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch, mode='func'):

        if mode == 'func':
            bf_x = inputs
            bs_x = other_x

        elif mode == 'struct':
            bs_x = inputs
            bf_x = other_x

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_for_captum_joint(self, bf_x, bs_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch):

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_return_embeddings(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch):
        """
        只返回中间节点嵌入（共享/私有）。
        不做池化、分类或损失计算。
        """
        # # # 1. 1*1卷积加权
        # x_func = bf_x.view(-1, 90, bf_x.shape[1]).permute(0,2,1)
        # x_func = self.conv_func(x_func)
        # bf_x = x_func.permute(0,2,1).reshape(-1, bf_x.shape[1])

        # x_struct = bs_x.view(-1, 90, bs_x.shape[1]).permute(0,2,1)
        # x_struct = self.conv_struct(x_struct)
        # bs_x = x_struct.permute(0,2,1).reshape(-1, bs_x.shape[1])

        # 2. 节点级编码
        emb_func = self.encoder_func(bf_x, bf_edge_index, bf_edge_weight, batch)   # [batch, nodes, hidden]
        emb_struct = self.encoder_struct(bs_x, bs_edge_index, bs_edge_weight, batch)

        # print("emb_func.shape:", emb_func.shape)
        # print("emb_struct.shape:", emb_struct.shape)

        # 3. 分解共享/私有嵌入
        zs_func, zp_func = self.head_func(emb_func)
        zs_struct, zp_struct = self.head_struct(emb_struct)

        ## 4. 可选跨模态注意力增强

        #emb_func, emb_struct = self.cross_attn_func_struct(emb_func, emb_struct)

        return {
            'emb_func': emb_func,
            'emb_struct': emb_struct,
            'zs_func': zs_func,
            'zs_struct': zs_struct,
            'zp_func': zp_func,
            'zp_struct': zp_struct
        }

    def forward(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, labels, batch, mode='train'):


        # 获取中间嵌入
        embeddings = self.forward_return_embeddings(bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch)

        emb_func = embeddings['emb_func']
        emb_struct = embeddings['emb_struct']
        zs_func = embeddings['zs_func']
        zs_struct = embeddings['zs_struct']
        zp_func = embeddings['zp_func']
        zp_struct = embeddings['zp_struct']

        emb_func = self.linear(emb_func)
        emb_struct = self.linear(emb_struct)

        batch_size = int(batch.max().item() + 1)
        emb_func = emb_func.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_func = emb_func.mean(dim=1)
        emb_struct = emb_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_struct = emb_struct.mean(dim=1)

        zs_func = zs_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_func = zs_func.mean(dim=1)
        zs_struct = zs_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_struct = zs_struct.mean(dim=1)

        zp_func = zp_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_func = zp_func.mean(dim=1)
        zp_struct = zp_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_struct = zp_struct.mean(dim=1)


        # --- 训练/测试逻辑 ---
        if mode == 'train':
            # 对比损失、正交损失
            contrastive_loss = bidirectional_contrastive_loss(zs_func, zs_struct, self.tau, self.lambda_param)
            ortho_loss = (orthogonal_loss(zs_func, zp_func)
                          + orthogonal_loss(zs_struct, zp_struct)
                          + orthogonal_loss(zp_func, zp_struct)) / 3


            enhanced_func = zs_func + self.alpha * zp_func
            enhanced_struct = zs_struct + self.alpha * zp_struct



            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)


            # 图级特征拼接、池化
            final_embedding = torch.cat([enhanced_func, enhanced_struct], dim=-1)

            # 分类
            logits = self.classifier_bn(final_embedding)

            return {
                'logits': logits,
                #'logits_spe': logits_spe,
                'contrastive_loss': contrastive_loss,
                'orthogonal_loss': ortho_loss,
                #'cluster_loss': cluster_loss,
            }

        elif mode == 'eval':
            # eval模式只做分类

            enhanced_func = zs_func + self.alpha * zp_func
            enhanced_struct = zs_struct + self.alpha * zp_struct

            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)

            final_embedding = torch.cat([fused_func, fused_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits

        elif mode == 'captum':
            #共享增强原始嵌入
            enhanced_func, attn_fs = self.enhancer(zp_func, zs_func, emb_func)

            # 结构模态：zp_struct 问，zs_struct 当 Key，emb_struct 当 Value
            enhanced_struct, attn_sf = self.enhancer(zp_struct, zs_struct, emb_struct)
            final_embedding = torch.cat([enhanced_func, enhanced_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits

class MultiModalGNN_new_new_new(nn.Module):
    """
    多模态图神经网络模型，集成了图编码、共享/私有嵌入分离和损失计算。

    流程:
    1. 三种模态数据 (fc, gm, fa) 分别通过独立的 GraphEncoder 得到图级表示。
    2. 每个图级表示通过 SharedPrivateHead 分解为共享嵌入和私有嵌入。
    3. 计算损失或进行分类:
        - 训练时，计算对比损失和正交损失。
        - 推理时，融合共享嵌入并进行分类。
    """
    def __init__(self, in_dim1, in_dim2, shared_dim, private_dim, num_classes, tau=0.07, lambda_param=0.5, hidden=4, learnable_loss_weights=False, norm='None'):
        super().__init__()
        if learnable_loss_weights:
            self.lcont = nn.Parameter(torch.tensor(1.0))
            self.lothro = nn.Parameter(torch.tensor(1.0))
        else:
            self.lcont = None
            self.lothro = None

        self.tau = tau
        self.lambda_param = lambda_param
        self.alpha = 0.5

        # 用来缓存固定的图结构
        self.edge_index_bf = None
        self.edge_weight_bf = None
        self.edge_index_bs = None
        self.edge_weight_bs = None
        self.batch = None

        #先用一个1*1卷积给拼接一起的不同特征分配一下权重
        self.conv_func = nn.Conv1d(in_channels=in_dim1, out_channels=in_dim1, kernel_size=1)
        self.conv_struct = nn.Conv1d(in_channels=in_dim2, out_channels=in_dim2, kernel_size=1)

        # 为每个模态创建一个图编码器
        # self.encoder_func = GATWithAttentionPooling(in_dim1, hidden, hidden)  # 功能模态
        # self.encoder_struct = GATWithAttentionPooling(in_dim2, hidden, hidden)    # 结构模态
        #两层
        # self.encoder_func = GraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        # self.encoder_struct = GraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)

        self.encoder_func = StableGraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        self.encoder_struct = StableGraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)
        # 输出维度
        self.hidden_dim = hidden



        # 创建共享/私有头
        if norm == 'None':
            self.head_func = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            # self.head_func = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)
            # self.head_struct = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)

        elif norm == 'ln':
            self.head_func = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
        elif norm == 'bn':
            self.head_func = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)

        # 跨模态注意力层
        #self.cross_attn_func_struct = CrossModalAttention_Bidirection(self.hidden_dim)

        #分层次跨模态融合
        #self.cross_attn_P_S_E = Fusion_P_S_E(self.hidden_dim, shared_dim, private_dim)

        # 将共享特征映射到与嵌入相同的维度，便于融合
        self.linear = nn.Linear(self.hidden_dim, shared_dim)# 将嵌入特征映射到与共享相同的维度
        # 让共享特征增强原始特征
        # self.cross_attn_S_E = CrossModalAttention_Mix(shared_dim)
        self.enhancer = CrossSpecAttention(d_model=shared_dim)

       # self.cluster_loss = ClusterLoss()

        # 分类器：输入为共享+私有2模态拼接
        #total_dim = shared_dim * 2 + private_dim * 2  # 1个共享 + 2个私有

        #self.conv1x1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.conv1x1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)


        #total_dim = shared_dim
        #total_dim = 2*self.hidden_dim
        total_dim = 2*shared_dim
        #total_dim = 4*(self.hidden_dim + 2)
        # 使用更复杂的分类器 - MLP with BatchNorm and Dropout
        self.classifier_bn = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.BatchNorm1d(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.BatchNorm1d(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_dim // 2, num_classes)
        )
        spe_dim = shared_dim
        self.classifier_spe = nn.Sequential(
            nn.Linear(spe_dim, spe_dim // 2),
            nn.BatchNorm1d(spe_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.LayerNorm(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(spe_dim // 2, num_classes)
        )

    def set_graph(self, bf_edge_index, bf_edge_weight,
                  bs_edge_index, bs_edge_weight, batch):
        """存储固定的图结构，供归因时使用"""
        self.edge_index_bf = bf_edge_index
        self.edge_weight_bf = bf_edge_weight
        self.edge_index_bs = bs_edge_index
        self.edge_weight_bs = bs_edge_weight
        self.batch = batch

    def mean_max_pool(self, x):
        mean_pool = x.mean(dim=1)   # [batch, feature]
        max_pool, _ = x.max(dim=1)  # [batch, feature]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2*feature]

    def _cross_attention(self, q, k, v, eps=1e-8):

        """
        简单的 scaled-dot-product cross-attention (按行计算)。
        q, k, v: (T, D)  # T = total_nodes
        返回与 q 同形状的 attended 输出。
        """
        # 注意：这里不做多头，仅做单头 scaled dot-product
        d = q.size(-1)
        scores = torch.matmul(q, k.t()) / math.sqrt(d)  # (T, T)
        attn = F.softmax(scores, dim=-1)  # 每行对所有 key 做归一
        out = torch.matmul(attn, v)  # (T, D)
        return out

    def forward_for_captum(self, inputs, other_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch, mode='func'):

        if mode == 'func':
            bf_x = inputs
            bs_x = other_x

        elif mode == 'struct':
            bs_x = inputs
            bf_x = other_x

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_for_captum_joint(self, bf_x, bs_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch):

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_return_embeddings(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch):
        """
        只返回中间节点嵌入（共享/私有）。
        不做池化、分类或损失计算。
        """
        # # # 1. 1*1卷积加权
        # x_func = bf_x.view(-1, 90, bf_x.shape[1]).permute(0,2,1)
        # x_func = self.conv_func(x_func)
        # bf_x = x_func.permute(0,2,1).reshape(-1, bf_x.shape[1])

        # x_struct = bs_x.view(-1, 90, bs_x.shape[1]).permute(0,2,1)
        # x_struct = self.conv_struct(x_struct)
        # bs_x = x_struct.permute(0,2,1).reshape(-1, bs_x.shape[1])

        # 2. 节点级编码
        emb_func = self.encoder_func(bf_x, bf_edge_index, bf_edge_weight, batch)   # [batch, nodes, hidden]
        emb_struct = self.encoder_struct(bs_x, bs_edge_index, bs_edge_weight, batch)

        # print("emb_func.shape:", emb_func.shape)
        # print("emb_struct.shape:", emb_struct.shape)

        # 3. 分解共享/私有嵌入
        zs_func, zp_func = self.head_func(emb_func)
        zs_struct, zp_struct = self.head_struct(emb_struct)

        ## 4. 可选跨模态注意力增强

        #emb_func, emb_struct = self.cross_attn_func_struct(emb_func, emb_struct)

        return {
            'emb_func': emb_func,
            'emb_struct': emb_struct,
            'zs_func': zs_func,
            'zs_struct': zs_struct,
            'zp_func': zp_func,
            'zp_struct': zp_struct
        }

    def forward(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, labels, batch, mode='train'):


        # 获取中间嵌入
        embeddings = self.forward_return_embeddings(bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch)

        emb_func = embeddings['emb_func']
        emb_struct = embeddings['emb_struct']
        zs_func = embeddings['zs_func']
        zs_struct = embeddings['zs_struct']
        zp_func = embeddings['zp_func']
        zp_struct = embeddings['zp_struct']


        batch_size = int(batch.max().item() + 1)
        emb_func = emb_func.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_func = emb_func.mean(dim=1)
        emb_struct = emb_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_struct = emb_struct.mean(dim=1)

        zs_func = zs_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_func = zs_func.mean(dim=1)
        zs_struct = zs_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_struct = zs_struct.mean(dim=1)

        zp_func = zp_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_func = zp_func.mean(dim=1)
        zp_struct = zp_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_struct = zp_struct.mean(dim=1)



        # --- 训练/测试逻辑 ---
        if mode == 'train':
            # === (1) 对比 & 正交 ===
            contrastive_loss = bidirectional_contrastive_loss(zs_func, zs_struct, self.tau, self.lambda_param)
            ortho_loss = (orthogonal_loss(zs_func, zp_func)
                        + orthogonal_loss(zs_struct, zp_struct)
                        + orthogonal_loss(zp_func, zp_struct)) / 3

            # === (2) Cross-attention 融合共享表示 ===
            zs_func_fused = self._cross_attention(zs_func, zs_struct, zs_struct)
            zs_struct_fused = self._cross_attention(zs_struct, zs_func, zs_func)

            # === (3) 私有特征辅助分类分支 ===
            logits_zp_func = self.classifier_spe(zp_func)
            logits_zp_struct = self.classifier_spe(zp_struct)

            ce_loss_func = F.cross_entropy(logits_zp_func, labels)
            ce_loss_struct = F.cross_entropy(logits_zp_struct, labels)
            spe_cls_loss = 0.5 * (ce_loss_func + ce_loss_struct)

            # === (4) 计算熵权重 (detach + 平滑) ===
            probs_zp_func = F.softmax(logits_zp_func, dim=1)
            probs_zp_struct = F.softmax(logits_zp_struct, dim=1)

            entropy_func = -torch.sum(probs_zp_func * torch.log(probs_zp_func + 1e-8), dim=1)
            entropy_struct = -torch.sum(probs_zp_struct * torch.log(probs_zp_struct + 1e-8), dim=1)

            max_entropy = math.log(probs_zp_func.size(1))
            weight_func = (1 - entropy_func / max_entropy).detach()
            weight_struct = (1 - entropy_struct / max_entropy).detach()

            # 平滑+残差混合 (共享主导)
            alpha = 0.4  # 私有特征影响比例
            beta = 0.3   # 防止完全为0
            weight_func_smooth = alpha * weight_func + beta
            weight_struct_smooth = alpha * weight_struct + beta

            # === (5) 共享主导残差融合 ===
            enhanced_func = zs_func_fused + weight_func_smooth.unsqueeze(1) * (0.5 * zp_func)
            enhanced_struct = zs_struct_fused + weight_struct_smooth.unsqueeze(1) * (0.5 * zp_struct)

            # === (6) 拼接 & 分类 ===
            final_node_repr = torch.cat([enhanced_func, enhanced_struct], dim=1)
            logits = self.classifier_bn(final_node_repr)

            # === (7) 熵正则（防止过度确定化） ===
            entropy_reg = -0.5 * (entropy_func.mean() + entropy_struct.mean())

            return {
                'logits': logits,
                'spe_cls_loss': spe_cls_loss,
                'contrastive_loss': contrastive_loss,
                'orthogonal_loss': ortho_loss,
                'entropy_reg': 0.01 * entropy_reg,
            }

        elif mode == 'eval':
            # === Cross-attention ===
            zs_func_fused = self._cross_attention(zs_func, zs_struct, zs_struct)
            zs_struct_fused = self._cross_attention(zs_struct, zs_func, zs_func)

            # === 私有置信度计算（detach） ===
            logits_zp_func = self.classifier_spe(zp_func)
            logits_zp_struct = self.classifier_spe(zp_struct)

            probs_zp_func = F.softmax(logits_zp_func, dim=1)
            probs_zp_struct = F.softmax(logits_zp_struct, dim=1)

            entropy_func = -torch.sum(probs_zp_func * torch.log(probs_zp_func + 1e-8), dim=1)
            entropy_struct = -torch.sum(probs_zp_struct * torch.log(probs_zp_struct + 1e-8), dim=1)
            max_entropy = math.log(probs_zp_func.size(1))
            weight_func = (1 - entropy_func / max_entropy).detach()
            weight_struct = (1 - entropy_struct / max_entropy).detach()

            alpha = 0.4
            beta = 0.3
            weight_func_smooth = alpha * weight_func + beta
            weight_struct_smooth = alpha * weight_struct + beta

            # === 残差融合 ===
            enhanced_func = zs_func_fused + weight_func_smooth.unsqueeze(1) * (0.5 * zp_func)
            enhanced_struct = zs_struct_fused + weight_struct_smooth.unsqueeze(1) * (0.5 * zp_struct)

            final_node_repr = torch.cat([enhanced_func, enhanced_struct], dim=1)
            logits = self.classifier_bn(final_node_repr)
            return logits


        elif mode == 'captum':
            #共享增强原始嵌入
            # stren_func = self.cross_attn_S_E(zp_func, emb_func) #zs_func是query
            # stren_struct = self.cross_attn_S_E(zp_struct, emb_struct) #zs_func是query
            # 功能模态：zp_func 问，zs_func 当 Key，emb_func 当 Value
            enhanced_func, attn_fs = self.enhancer(zp_func, zs_func, emb_func)

            # 结构模态：zp_struct 问，zs_struct 当 Key，emb_struct 当 Value
            enhanced_struct, attn_sf = self.enhancer(zp_struct, zs_struct, emb_struct)
            final_embedding = torch.cat([enhanced_func, enhanced_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits

def freeze_by_names(model, module_name_list):
    """冻结 model 中名字包含 module_name_list 中任一字符串的 module 参数"""
    for name, param in model.named_parameters():
        param.requires_grad = True  # default enable
    for name, param in model.named_parameters():
        for mn in module_name_list:
            if mn in name:
                param.requires_grad = False
                break

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MultiModalGNN_new_new_new_new(nn.Module):
    """
    多模态图神经网络模型，集成了图编码、共享/私有嵌入分离和损失计算。

    流程:
    1. 三种模态数据 (fc, gm, fa) 分别通过独立的 GraphEncoder 得到图级表示。
    2. 每个图级表示通过 SharedPrivateHead 分解为共享嵌入和私有嵌入。
    3. 计算损失或进行分类:
        - 训练时，计算对比损失和正交损失。
        - 推理时，融合共享嵌入并进行分类。
    """
    def __init__(self, in_dim1, in_dim2, aux_dim, shared_dim, private_dim, num_classes, tau=0.07, lambda_param=0.5, hidden=4, learnable_loss_weights=False, norm='None', dropout=0.1):
        super().__init__()
        if learnable_loss_weights:
            self.lcont = nn.Parameter(torch.tensor(1.0))
            self.lothro = nn.Parameter(torch.tensor(1.0))
        else:
            self.lcont = None
            self.lothro = None

        self.tau = tau
        self.lambda_param = lambda_param
        self.alpha = 0.5

        # 用来缓存固定的图结构
        self.edge_index_bf = None
        self.edge_weight_bf = None
        self.edge_index_bs = None
        self.edge_weight_bs = None
        self.batch = None

        #先用一个1*1卷积给拼接一起的不同特征分配一下权重
        self.conv_func = nn.Conv1d(in_channels=in_dim1, out_channels=in_dim1, kernel_size=1)
        self.conv_struct = nn.Conv1d(in_channels=in_dim2, out_channels=in_dim2, kernel_size=1)

        # 为每个模态创建一个图编码器
        # self.encoder_func = GATWithAttentionPooling(in_dim1, hidden, hidden)  # 功能模态
        # self.encoder_struct = GATWithAttentionPooling(in_dim2, hidden, hidden)    # 结构模态
        #两层
        # self.encoder_func = GraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        # self.encoder_struct = GraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)

        self.encoder_func = StableGraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        self.encoder_struct = StableGraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)
        # 输出维度
        self.hidden_dim = hidden



        # 创建共享/私有头
        if norm == 'None':
            self.head_func = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            # self.head_func = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)
            # self.head_struct = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)

        elif norm == 'ln':
            self.head_func = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
        elif norm == 'bn':
            self.head_func = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)

        # 跨模态注意力层
        #self.cross_attn_func_struct = CrossModalAttention_Bidirection(self.hidden_dim)

        #分层次跨模态融合
        #self.cross_attn_P_S_E = Fusion_P_S_E(self.hidden_dim, shared_dim, private_dim)

        # 将共享特征映射到与嵌入相同的维度，便于融合
        self.linear = nn.Linear(self.hidden_dim, shared_dim)# 将嵌入特征映射到与共享相同的维度

        self.norm_func_cat = nn.LayerNorm(shared_dim + aux_dim)
        self.norm_struct_cat = nn.LayerNorm(shared_dim + aux_dim)
        #加入另外的模态的维度映射
        self.fuse_struct = nn.Linear(shared_dim + aux_dim, shared_dim)
        self.fuse_func   = nn.Linear(shared_dim + aux_dim, shared_dim)


        # 让共享特征增强原始特征
        self.cross_attn_S_E = CrossModalAttention_Mix(shared_dim)
        self.enhancer = CrossSpecAttention(d_model=shared_dim)

       # self.cluster_loss = ClusterLoss()

        # 分类器：输入为共享+私有2模态拼接
        #total_dim = shared_dim * 2 + private_dim * 2  # 1个共享 + 2个私有

        #self.conv1x1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.conv1x1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)


        #total_dim = shared_dim
        #total_dim = 2*self.hidden_dim
        total_dim = 2*shared_dim
        #total_dim = 4*(self.hidden_dim + 2)
        # 使用更复杂的分类器 - MLP with BatchNorm and Dropout
        self.classifier_bn = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.BatchNorm1d(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.BatchNorm1d(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_dim // 2, num_classes)
        )
        spe_dim = 1*shared_dim
        self.classifier_spe = nn.Sequential(
            nn.Linear(spe_dim, spe_dim // 2),
            nn.BatchNorm1d(spe_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.LayerNorm(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(spe_dim // 2, num_classes)
        )

        # ---- Learnable Gate for each modality ----
        self.gate_mlp_func = nn.Sequential(
            nn.Linear(num_classes + shared_dim, 16),  # 输入是 [分类概率 + 私有特征]
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.gate_mlp_struct = nn.Sequential(
            nn.Linear(num_classes + shared_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # 初始化偏置为负，初始时私有影响较弱
        with torch.no_grad():
            self.gate_mlp_func[-1].bias.data.fill_(-1.0)
            self.gate_mlp_struct[-1].bias.data.fill_(-1.0)

        self.t1_encoder = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, aux_dim)
        )

        self.asl_encoder = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, aux_dim)
        )

    def set_graph(self, bf_edge_index, bf_edge_weight,
                  bs_edge_index, bs_edge_weight, batch):
        """存储固定的图结构，供归因时使用"""
        self.edge_index_bf = bf_edge_index
        self.edge_weight_bf = bf_edge_weight
        self.edge_index_bs = bs_edge_index
        self.edge_weight_bs = bs_edge_weight
        self.batch = batch

    def mean_max_pool(self, x):
        mean_pool = x.mean(dim=1)   # [batch, feature]
        max_pool, _ = x.max(dim=1)  # [batch, feature]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2*feature]

    def freeze_main_encoders(self):
        freeze_by_names(self, ['encoder_func', 'encoder_struct', 'head_func', 'head_struct',
                               'gate_mlp_func', 'gate_mlp_struct', 'enhancer', 'classifier_bn'])
        print(f"Freeze main encoders, trainable params: {count_trainable_params(self)}")

    def freeze_t1_asl(self):
        if hasattr(self, 't1_encoder'):
            for p in self.t1_encoder.parameters():
                p.requires_grad = False
        if hasattr(self, 'asl_encoder'):
            for p in self.asl_encoder.parameters():
                p.requires_grad = False
        if hasattr(self, 'fuse_struct'):
            for p in self.fuse_struct.parameters():
                p.requires_grad = False
        if hasattr(self, 'fuse_func'):
            for p in self.fuse_func.parameters():
                p.requires_grad = False
        if hasattr(self, 'norm_func_cat'):
            for p in self.norm_func_cat.parameters():
                p.requires_grad = False
        if hasattr(self, 'norm_struct_cat'):
            for p in self.norm_struct_cat.parameters():
                p.requires_grad = False
        print(f"Freeze T1/ASL encoders, trainable params: {count_trainable_params(self)}")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print(f"Unfreeze all params: {count_trainable_params(self)}")

    def forward_for_captum(self, inputs, other_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch, mode='func'):

        if mode == 'func':
            bf_x = inputs
            bs_x = other_x

        elif mode == 'struct':
            bs_x = inputs
            bf_x = other_x

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_for_captum_joint(self, bf_x, bs_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch):

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_return_embeddings(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch):
        """
        只返回中间节点嵌入（共享/私有）。
        不做池化、分类或损失计算。
        """
        # # # 1. 1*1卷积加权
        # x_func = bf_x.view(-1, 90, bf_x.shape[1]).permute(0,2,1)
        # x_func = self.conv_func(x_func)
        # bf_x = x_func.permute(0,2,1).reshape(-1, bf_x.shape[1])

        # x_struct = bs_x.view(-1, 90, bs_x.shape[1]).permute(0,2,1)
        # x_struct = self.conv_struct(x_struct)
        # bs_x = x_struct.permute(0,2,1).reshape(-1, bs_x.shape[1])

        # 2. 节点级编码
        emb_func = self.encoder_func(bf_x, bf_edge_index, bf_edge_weight, batch)   # [batch, nodes, hidden]
        emb_struct = self.encoder_struct(bs_x, bs_edge_index, bs_edge_weight, batch)

        # print("emb_func.shape:", emb_func.shape)
        # print("emb_struct.shape:", emb_struct.shape)

        # 3. 分解共享/私有嵌入
        zs_func, zp_func = self.head_func(emb_func)
        zs_struct, zp_struct = self.head_struct(emb_struct)


        return {
            'emb_func': emb_func,
            'emb_struct': emb_struct,
            'zs_func': zs_func,
            'zs_struct': zs_struct,
            'zp_func': zp_func,
            'zp_struct': zp_struct
        }

    def forward(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, labels, batch, t1_input, asl_input, mode='train'):


        # 获取中间嵌入
        embeddings = self.forward_return_embeddings(bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch)

        emb_func = embeddings['emb_func']
        emb_struct = embeddings['emb_struct']
        zs_func = embeddings['zs_func']
        zs_struct = embeddings['zs_struct']
        zp_func = embeddings['zp_func']
        zp_struct = embeddings['zp_struct']

        emb_func = self.linear(emb_func)
        emb_struct = self.linear(emb_struct)

        batch_size = int(batch.max().item() + 1)
        emb_func = emb_func.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_func = emb_func.mean(dim=1)
        emb_struct = emb_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_struct = emb_struct.mean(dim=1)

        zs_func = zs_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_func = zs_func.mean(dim=1)
        zs_struct = zs_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_struct = zs_struct.mean(dim=1)

        zp_func = zp_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_func = zp_func.mean(dim=1)
        zp_struct = zp_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_struct = zp_struct.mean(dim=1)


        # --- 训练/测试逻辑 ---
        if mode == 'train':
            # 对比损失、正交损失
            contrastive_loss = bidirectional_contrastive_loss(zs_func, zs_struct, self.tau, self.lambda_param)
            ortho_loss = (orthogonal_loss(zs_func, zp_func)
                          + orthogonal_loss(zs_struct, zp_struct)
                          + orthogonal_loss(zp_func, zp_struct)) / 3


           # === (3) 私有特征辅助分类分支 ===
            logits_zp_func = self.classifier_spe(zp_func)
            logits_zp_struct = self.classifier_spe(zp_struct)

            ce_loss_func = F.cross_entropy(logits_zp_func, labels)
            ce_loss_struct = F.cross_entropy(logits_zp_struct, labels)
            spe_cls_loss = 0.5 * (ce_loss_func + ce_loss_struct)

            # === (4) 计算概率 ===
            probs_zp_func = F.softmax(logits_zp_func, dim=1)
            probs_zp_struct = F.softmax(logits_zp_struct, dim=1)

            # === (5) Gate 计算 ===
            # 输入 = [分类概率 + 私有特征]，输出 gate ∈ (0,1)
            gate_input_func = torch.cat([probs_zp_func.detach(), zp_func], dim=1)
            gate_input_struct = torch.cat([probs_zp_struct.detach(), zp_struct], dim=1)

            gate_func = torch.sigmoid(self.gate_mlp_func(gate_input_func))
            gate_struct = torch.sigmoid(self.gate_mlp_struct(gate_input_struct))


            # === (6) 残差融合 ===
            enhanced_func = zs_func + gate_func * zp_func * 0.4
            enhanced_struct = zs_struct + gate_struct * zp_struct * 0.4

            # === (6.5) T1 / ASL 编码与调制 ===
            if t1_input is not None and asl_input is not None:
                t1_feat = self.t1_encoder(t1_input)
                asl_feat = self.asl_encoder(asl_input)

                # 对 ROI / 时间步维度取平均
                emb_t1 = t1_feat.mean(dim=1)
                emb_asl = asl_feat.mean(dim=1)

                # 拼接
                emb_func_cat = torch.cat([emb_func, emb_asl], dim=-1)
                emb_struct_cat = torch.cat([emb_struct, emb_t1], dim=-1)

                # 新线性层
                emb_func = self.fuse_func(self.norm_func_cat(emb_func_cat))
                emb_struct = self.fuse_struct(self.norm_struct_cat(emb_struct_cat))

            # === (7) 后续 cross-modal 融合 ===
            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)

            final_embedding = torch.cat([fused_func, fused_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)

            return {
                'logits': logits,
                'spe_cls_loss': spe_cls_loss,
                'contrastive_loss': contrastive_loss,
                'orthogonal_loss': ortho_loss,
                #'cluster_loss': cluster_loss,
            }

        elif mode == 'eval':
            # eval模式只做分类

            # === 私有置信度计算（detach） ===
            logits_zp_func = self.classifier_spe(zp_func)
            logits_zp_struct = self.classifier_spe(zp_struct)

            # === (4) 计算概率 ===
            probs_zp_func = F.softmax(logits_zp_func, dim=1)
            probs_zp_struct = F.softmax(logits_zp_struct, dim=1)

            # === (5) Gate 计算 ===
            # 输入 = [分类概率 + 私有特征]，输出 gate ∈ (0,1)
            gate_input_func = torch.cat([probs_zp_func.detach(), zp_func], dim=1)
            gate_input_struct = torch.cat([probs_zp_struct.detach(), zp_struct], dim=1)

            gate_func = torch.sigmoid(self.gate_mlp_func(gate_input_func))
            gate_struct = torch.sigmoid(self.gate_mlp_struct(gate_input_struct))

            # === (6) 残差融合 ===
            enhanced_func = zs_func + gate_func * zp_func * 0.4
            enhanced_struct = zs_struct + gate_struct * zp_struct * 0.4

            # === (6.5) T1 / ASL 编码与调制 ===
            if t1_input is not None and asl_input is not None:
                t1_feat = self.t1_encoder(t1_input)
                asl_feat = self.asl_encoder(asl_input)

                # 对 ROI / 时间步维度取平均
                emb_t1 = t1_feat.mean(dim=1)
                emb_asl = asl_feat.mean(dim=1)

                # 拼接
                emb_func_cat = torch.cat([emb_func, emb_asl], dim=-1)
                emb_struct_cat = torch.cat([emb_struct, emb_t1], dim=-1)

                # 新线性层
                emb_func = self.fuse_func(self.norm_func_cat(emb_func_cat))
                emb_struct = self.fuse_struct(self.norm_struct_cat(emb_struct_cat))


            # === (7) 后续 cross-modal 融合 ===
            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)

            final_embedding = torch.cat([fused_func, fused_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits

        elif mode == 'captum':
            #共享增强原始嵌入
            enhanced_func, attn_fs = self.enhancer(zp_func, zs_func, emb_func)

            # 结构模态：zp_struct 问，zs_struct 当 Key，emb_struct 当 Value
            enhanced_struct, attn_sf = self.enhancer(zp_struct, zs_struct, emb_struct)
            final_embedding = torch.cat([enhanced_func, enhanced_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits

class MultiModalGNN_new_new_new_new_new(nn.Module):
    """
    多模态图神经网络模型，集成了图编码、共享/私有嵌入分离和损失计算。

    流程:
    1. 三种模态数据 (fc, gm, fa) 分别通过独立的 GraphEncoder 得到图级表示。
    2. 每个图级表示通过 SharedPrivateHead 分解为共享嵌入和私有嵌入。
    3. 计算损失或进行分类:
        - 训练时，计算对比损失和正交损失。
        - 推理时，融合共享嵌入并进行分类。
    """
    def __init__(self, in_dim1, in_dim2, aux_dim, shared_dim, private_dim, num_classes, tau=0.07, lambda_param=0.5, hidden=4, learnable_loss_weights=False, norm='None'):
        super().__init__()
        if learnable_loss_weights:
            self.lcont = nn.Parameter(torch.tensor(1.0))
            self.lothro = nn.Parameter(torch.tensor(1.0))
        else:
            self.lcont = None
            self.lothro = None

        self.tau = tau
        self.lambda_param = lambda_param
        self.alpha = 0.1

        # 用来缓存固定的图结构
        self.edge_index_bf = None
        self.edge_weight_bf = None
        self.edge_index_bs = None
        self.edge_weight_bs = None
        self.batch = None
        self.t1 = None
        self.asl = None

        #先用一个1*1卷积给拼接一起的不同特征分配一下权重
        self.conv_func = nn.Conv1d(in_channels=in_dim1, out_channels=in_dim1, kernel_size=1)
        self.conv_struct = nn.Conv1d(in_channels=in_dim2, out_channels=in_dim2, kernel_size=1)

        # 为每个模态创建一个图编码器
        # self.encoder_func = GATWithAttentionPooling(in_dim1, hidden, hidden)  # 功能模态
        # self.encoder_struct = GATWithAttentionPooling(in_dim2, hidden, hidden)    # 结构模态
        #两层
        # self.encoder_func = GraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        # self.encoder_struct = GraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)

        self.encoder_func = StableGraphEncoder(in_dim1, hidden)  # 功能模态  (b, nodes, hidden)
        self.encoder_struct = StableGraphEncoder(in_dim2, hidden)    # 结构模态  (b, nodes, hidden)
        # 输出维度
        self.hidden_dim = hidden



        # 创建共享/私有头
        if norm == 'None':
            self.head_func = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead(self.hidden_dim, shared_dim, private_dim)
            # self.head_func = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)
            # self.head_struct = SharedPrivateHead_pool(self.hidden_dim, shared_dim, private_dim)

        elif norm == 'ln':
            self.head_func = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_LN(self.hidden_dim, shared_dim, private_dim)
        elif norm == 'bn':
            self.head_func = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)
            self.head_struct = SharedPrivateHead_BN(self.hidden_dim, shared_dim, private_dim)

        # 跨模态注意力层
        #self.cross_attn_func_struct = CrossModalAttention_Bidirection(self.hidden_dim)

        #分层次跨模态融合
        #self.cross_attn_P_S_E = Fusion_P_S_E(self.hidden_dim, shared_dim, private_dim)

        # 将共享特征映射到与嵌入相同的维度，便于融合
        self.linear = nn.Linear(self.hidden_dim, shared_dim)# 将嵌入特征映射到与共享相同的维度

        # self.norm_func_cat = nn.LayerNorm(shared_dim + aux_dim)
        # self.norm_struct_cat = nn.LayerNorm(shared_dim + aux_dim)
        #加入另外的模态的维度映射
        self.fuse_struct = nn.Linear(hidden + aux_dim, hidden)
        self.fuse_func   = nn.Linear(hidden + aux_dim, hidden)


        # 让共享特征增强原始特征
        self.cross_attn_S_E = CrossModalAttention_Mix(shared_dim)
        self.enhancer = CrossSpecAttention(d_model=shared_dim)

       # self.cluster_loss = ClusterLoss()

        # 分类器：输入为共享+私有2模态拼接
        #total_dim = shared_dim * 2 + private_dim * 2  # 1个共享 + 2个私有

        #self.conv1x1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
        self.conv1x1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)


        #total_dim = shared_dim
        #total_dim = 2*self.hidden_dim
        total_dim = 2*shared_dim
        #total_dim = 4*(self.hidden_dim + 2)
        # 使用更复杂的分类器 - MLP with BatchNorm and Dropout
        self.classifier_bn = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.BatchNorm1d(total_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.BatchNorm1d(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(total_dim // 2, num_classes)
        )
        spe_dim = 1*shared_dim
        self.classifier_spe = nn.Sequential(
            nn.Linear(spe_dim, spe_dim // 2),
            nn.BatchNorm1d(spe_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(total_dim // 2, total_dim // 4),
            # nn.LayerNorm(total_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(spe_dim // 2, num_classes)
        )

        # ---- Learnable Gate for each modality ----
        self.gate_mlp_func = nn.Sequential(
            nn.Linear(num_classes + shared_dim, 16),  # 输入是 [分类概率 + 私有特征]
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.gate_mlp_struct = nn.Sequential(
            nn.Linear(num_classes + shared_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.gate_func = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.gate_struct = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )


        # 初始化偏置为负，初始时私有影响较弱
        with torch.no_grad():
            self.gate_mlp_func[-1].bias.data.fill_(-1.0)
            self.gate_mlp_struct[-1].bias.data.fill_(-1.0)

        self.t1_encoder = nn.Sequential(
            nn.Linear(1, aux_dim//2),
            nn.ReLU(),
            nn.Linear(aux_dim//2, aux_dim),
            nn.BatchNorm1d(aux_dim),  # ✅ 对齐主模态分布
            nn.Dropout(0.2),

        )

        self.asl_encoder = nn.Sequential(
            nn.Linear(1, aux_dim//2),
            nn.ReLU(),
            nn.Linear(aux_dim//2, aux_dim),
            nn.BatchNorm1d(aux_dim),  # ✅ 对齐主模态分布
            nn.Dropout(0.2),
        )


        # Direct auxiliary classifiers give T1/ASL explicit diagnosis
        # supervision instead of relying only on the correlation penalty.
        self.classifier_t1 = nn.Linear(aux_dim, num_classes)
        self.classifier_asl = nn.Linear(aux_dim, num_classes)

    def set_graph(self, bf_edge_index, bf_edge_weight,
                  bs_edge_index, bs_edge_weight, batch, t1, asl):
        """存储固定的图结构，供归因时使用"""
        self.edge_index_bf = bf_edge_index
        self.edge_weight_bf = bf_edge_weight
        self.edge_index_bs = bs_edge_index
        self.edge_weight_bs = bs_edge_weight
        self.batch = batch
        self.t1 = t1
        self.asl = asl

    def mean_max_pool(self, x):
        mean_pool = x.mean(dim=1)   # [batch, feature]
        max_pool, _ = x.max(dim=1)  # [batch, feature]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [batch, 2*feature]

    def freeze_main_encoders(self):
        freeze_by_names(self, ['encoder_func', 'encoder_struct', 'head_func', 'head_struct',
                               'gate_mlp_func', 'gate_mlp_struct', 'enhancer', 'classifier_bn'])
        print(f"Freeze main encoders, trainable params: {count_trainable_params(self)}")

    def freeze_t1_asl(self):
        if hasattr(self, 't1_encoder'):
            for p in self.t1_encoder.parameters():
                p.requires_grad = False
        if hasattr(self, 'asl_encoder'):
            for p in self.asl_encoder.parameters():
                p.requires_grad = False
        if hasattr(self, 'gate_struct'):
            for p in self.gate_struct.parameters():
                p.requires_grad = False
        if hasattr(self, 'gate_func'):
            for p in self.gate_func.parameters():
                p.requires_grad = False
        # if hasattr(self, 'norm_func_cat'):
        #     for p in self.norm_func_cat.parameters():
        #         p.requires_grad = False
        # if hasattr(self, 'norm_struct_cat'):
        #     for p in self.norm_struct_cat.parameters():
        #         p.requires_grad = False
        print(f"Freeze T1/ASL encoders, trainable params: {count_trainable_params(self)}")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print(f"Unfreeze all params: {count_trainable_params(self)}")

    def forward_for_captum(self, inputs, other_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch, mode='func'):

        if mode == 'func':
            bf_x = inputs
            bs_x = other_x

        elif mode == 'struct':
            bs_x = inputs
            bf_x = other_x

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]

    def forward_for_captum_joint(self, bf_x, bs_x,
                       bf_edge_index, bf_edge_weight,
                       bs_edge_index, bs_edge_weight,
                       labels,
                       batch):

        out = self.forward(
            bf_x, self.edge_index_bf, self.edge_weight_bf,
            bs_x, self.edge_index_bs, self.edge_weight_bs,
            labels, batch, mode='captum'
        )

        return out[:, 1]
    def forward_for_captum_node(
        self, bf_x, bs_x,
        bf_edge_index, bf_edge_weight,
        bs_edge_index, bs_edge_weight,
        labels, batch, t1_input=None, asl_input=None, mode='func'
    ):
        """
        节点级 forward_for_captum，用于 Captum 归因。
        返回每个节点的 embedding 贡献，同时保留 cross-modal 融合和 classifier。
        """

        # 按模态选择输入
        if mode == 'func':
            bf_input = bf_x
            bs_input = bs_x
        elif mode == 'struct':
            bf_input = bs_x
            bs_input = bf_x

        print(bf_input.shape)
        print(bf_edge_index.shape)
        print(bf_edge_weight.shape)
        print(self.edge_index_bf.shape)
        print(self.edge_weight_bf.shape)
        # embeddings = self.forward_return_embeddings(
        #     bf_input, bf_edge_index, bf_edge_weight,
        #     bs_input, bs_edge_index, bs_edge_weight,
        #     batch
        # )
        embeddings = self.forward_return_embeddings(
            bf_input, self.edge_index_bf, self.edge_weight_bf,
            bs_input, self.edge_index_bs, self.edge_weight_bs,
            self.batch
        )

        N = 90
        B = bf_input.shape[0] // N  # 节点级 batch_size

        # 节点级 embedding
        emb_func = embeddings['emb_func']#.reshape(B, N, -1)
        emb_struct = embeddings['emb_struct']#.reshape(B, N, -1)
        zs_func = embeddings['zs_func']#.reshape(B, N, -1)
        zs_struct = embeddings['zs_struct']#.reshape(B, N, -1)
        zp_func = embeddings['zp_func']#.reshape(B, N, -1)
        zp_struct = embeddings['zp_struct']#.reshape(B, N, -1)

        # 私有特征辅助分类

        logits_zp_func = self.classifier_spe(zp_func)  # [B*N, num_classes]
        logits_zp_struct = self.classifier_spe(zp_struct)

        probs_zp_func = F.softmax(logits_zp_func, dim=-1)
        probs_zp_struct = F.softmax(logits_zp_struct, dim=-1)

        # Gate 计算
        gate_func = torch.sigmoid(self.gate_mlp_func(torch.cat([probs_zp_func.detach(), zp_func], dim=-1)))
        gate_struct = torch.sigmoid(self.gate_mlp_struct(torch.cat([probs_zp_struct.detach(), zp_struct], dim=-1)))

        # 残差融合，节点级
        enhanced_func = zs_func + gate_func * zp_func * 0.4
        enhanced_struct = zs_struct + gate_struct * zp_struct * 0.4

        # T1 / ASL 融合
        if t1_input is not None and asl_input is not None:
            emb_t1 = self.t1_encoder(t1_input.reshape(B * N, t1_input.shape[-1]))#.reshape(B, N, -1)
            emb_asl = self.asl_encoder(asl_input.reshape(B * N, asl_input.shape[-1]))#.reshape(B, N, -1)
            gate_func_extra = torch.sigmoid(self.gate_func(torch.cat([emb_func, emb_asl], dim=-1)))
            gate_struct_extra = torch.sigmoid(self.gate_struct(torch.cat([emb_struct, emb_t1], dim=-1)))
            emb_func = emb_func + gate_func_extra * emb_asl
            emb_struct = emb_struct + gate_struct_extra * emb_t1

        # Linear 投影
        emb_func = self.linear(emb_func)
        emb_struct = self.linear(emb_struct)

        # Cross-modal 融合，节点级
        fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
        fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)

        # 节点级 final embedding
        final_node_embedding = torch.cat([fused_func, fused_struct], dim=-1)  # [B* N, F_total]

        # 节点级 logits
        node_logits = self.classifier_bn(final_node_embedding)  # [B* N, num_classes]
        print('node_logits:', node_logits.shape)
        # node_logits_flat = node_logits.view(-1, node_logits.shape[-1])  # [B*N, C]
        return node_logits

    def forward_return_embeddings(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch):
        """
        只返回中间节点嵌入（共享/私有）。
        不做池化、分类或损失计算。
        """
        # # # 1. 1*1卷积加权
        # x_func = bf_x.view(-1, 90, bf_x.shape[1]).permute(0,2,1)
        # x_func = self.conv_func(x_func)
        # bf_x = x_func.permute(0,2,1).reshape(-1, bf_x.shape[1])

        # x_struct = bs_x.view(-1, 90, bs_x.shape[1]).permute(0,2,1)
        # x_struct = self.conv_struct(x_struct)
        # bs_x = x_struct.permute(0,2,1).reshape(-1, bs_x.shape[1])

        # 2. 节点级编码
        emb_func = self.encoder_func(bf_x, bf_edge_index, bf_edge_weight, batch)   # [batch, nodes, hidden]
        emb_struct = self.encoder_struct(bs_x, bs_edge_index, bs_edge_weight, batch)

        # print("emb_func.shape:", emb_func.shape)
        # print("emb_struct.shape:", emb_struct.shape)

        # 3. 分解共享/私有嵌入
        zs_func, zp_func = self.head_func(emb_func)
        zs_struct, zp_struct = self.head_struct(emb_struct)


        return {
            'emb_func': emb_func,
            'emb_struct': emb_struct,
            'zs_func': zs_func,
            'zs_struct': zs_struct,
            'zp_func': zp_func,
            'zp_struct': zp_struct
        }

    def forward(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, labels, batch, t1_input, asl_input, mode='train'):


        # 获取中间嵌入
        embeddings = self.forward_return_embeddings(bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch)
        batch_size = int(batch.max().item() + 1)

        emb_func = embeddings['emb_func']
        emb_struct = embeddings['emb_struct']
        zs_func = embeddings['zs_func']
        zs_struct = embeddings['zs_struct']
        zp_func = embeddings['zp_func']
        zp_struct = embeddings['zp_struct']

        emb_func = emb_func.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_func = emb_func.mean(dim=1)
        emb_struct = emb_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        emb_struct = emb_struct.mean(dim=1)

        zs_func = zs_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_func = zs_func.mean(dim=1)
        zs_struct = zs_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zs_struct = zs_struct.mean(dim=1)

        zp_func = zp_func.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_func = zp_func.mean(dim=1)
        zp_struct = zp_struct.reshape(batch_size, 90, -1)  # (B, N, F)
        zp_struct = zp_struct.mean(dim=1)

        #安全
        aux_loss = torch.tensor(0.0, device=emb_func.device)
        aux_cls_loss = torch.tensor(0.0, device=emb_func.device)
        emb_asl, emb_t1 = None, None
        # === (6.5) T1 / ASL 编码与调制 ===
        if t1_input is not None and asl_input is not None:
            B, N, f = t1_input.shape

            # --- T1 ---
            t1_feat = t1_input.reshape(B * N, f)
            t1_feat = self.t1_encoder(t1_feat)
            t1_feat = t1_feat.reshape(B, N, -1)

            # --- ASL ---
            asl_feat = asl_input.reshape(B * N, f)
            asl_feat = self.asl_encoder(asl_feat)
            asl_feat = asl_feat.reshape(B, N, -1)

            # --- 池化 ---
            emb_t1 = t1_feat.mean(dim=1)
            emb_asl = asl_feat.mean(dim=1)


        # --- 训练/测试逻辑 ---
        if mode == 'train':
            # 对比损失、正交损失
            contrastive_loss = bidirectional_contrastive_loss(zs_func, zs_struct, self.tau, self.lambda_param)
            ortho_loss = (orthogonal_loss(zs_func, zp_func)
                          + orthogonal_loss(zs_struct, zp_struct)
                          + orthogonal_loss(zp_func, zp_struct)) / 3


            # === (3) 私有特征辅助分类分支 ===
            logits_zp_func = self.classifier_spe(zp_func)
            logits_zp_struct = self.classifier_spe(zp_struct)

            ce_loss_func = F.cross_entropy(logits_zp_func, labels)
            ce_loss_struct = F.cross_entropy(logits_zp_struct, labels)
            spe_cls_loss = 0.5 * (ce_loss_func + ce_loss_struct)

            # === (4) 计算概率 ===
            probs_zp_func = F.softmax(logits_zp_func, dim=1)
            probs_zp_struct = F.softmax(logits_zp_struct, dim=1)

            # === (5) Gate 计算 ===
            # 输入 = [分类概率 + 私有特征]，输出 gate ∈ (0,1)
            gate_input_func = torch.cat([probs_zp_func.detach(), zp_func], dim=1)
            gate_input_struct = torch.cat([probs_zp_struct.detach(), zp_struct], dim=1)

            gate_func = torch.sigmoid(self.gate_mlp_func(gate_input_func))
            gate_struct = torch.sigmoid(self.gate_mlp_struct(gate_input_struct))


            # === (6) 残差融合 ===
            enhanced_func = zs_func + gate_func * zp_func * 0.4
            enhanced_struct = zs_struct + gate_struct * zp_struct * 0.4

            if t1_input is not None and asl_input is not None:
                #去冗余
                loss_corr_asl = correlation_penalty(emb_func.detach(), emb_asl)
                loss_corr_t1  = correlation_penalty(emb_struct.detach(), emb_t1)
                aux_loss = loss_corr_asl + loss_corr_t1
                aux_cls_loss = 0.5 * (
                    F.cross_entropy(self.classifier_t1(emb_t1), labels)
                    + F.cross_entropy(self.classifier_asl(emb_asl), labels)
                )

                # 拼接
                gate_func = torch.sigmoid(self.gate_func(torch.cat([emb_func, emb_asl], dim=-1)))
                emb_func = emb_func + gate_func * emb_asl

                gate_struct = torch.sigmoid(self.gate_struct(torch.cat([emb_struct, emb_t1], dim=-1)))
                emb_struct = emb_struct + gate_struct * emb_t1

                # emb_func = emb_func + self.alpha * emb_asl
                # emb_struct = emb_struct + self.alpha * emb_t1

            emb_func = self.linear(emb_func)
            emb_struct = self.linear(emb_struct)


            # === (7) 后续 cross-modal 融合 ===
            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)


            final_embedding = torch.cat([fused_func, fused_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)

            return {
                'logits': logits,
                'spe_cls_loss': spe_cls_loss,
                'contrastive_loss': contrastive_loss,
                'orthogonal_loss': ortho_loss,
                'aux_loss': aux_loss,
                'aux_cls_loss': aux_cls_loss,
                #'cluster_loss': cluster_loss,
            }

        elif mode == 'eval':
            # eval模式只做分类

            # === 私有置信度计算（detach） ===
            logits_zp_func = self.classifier_spe(zp_func)
            logits_zp_struct = self.classifier_spe(zp_struct)

            # === (4) 计算概率 ===
            probs_zp_func = F.softmax(logits_zp_func, dim=1)
            probs_zp_struct = F.softmax(logits_zp_struct, dim=1)

            # === (5) Gate 计算 ===
            # 输入 = [分类概率 + 私有特征]，输出 gate ∈ (0,1)
            gate_input_func = torch.cat([probs_zp_func.detach(), zp_func], dim=1)
            gate_input_struct = torch.cat([probs_zp_struct.detach(), zp_struct], dim=1)

            gate_func = torch.sigmoid(self.gate_mlp_func(gate_input_func))
            gate_struct = torch.sigmoid(self.gate_mlp_struct(gate_input_struct))

            # === (6) 残差融合 ===
            enhanced_func = zs_func + gate_func * zp_func * 0.4
            enhanced_struct = zs_struct + gate_struct * zp_struct * 0.4

            # Evaluation-only graph-modality ablation. The auxiliary ASL/T1
            # injection below remains active, so only the requested graph
            # modality is removed.
            if getattr(self, '_ablate_functional_graph', False):
                enhanced_func = torch.zeros_like(enhanced_func)
                emb_func = torch.zeros_like(emb_func)
            if getattr(self, '_ablate_structural_graph', False):
                enhanced_struct = torch.zeros_like(enhanced_struct)
                emb_struct = torch.zeros_like(emb_struct)

            if t1_input is not None and asl_input is not None:
                # 拼接
                # emb_func = emb_func + self.alpha * emb_asl
                # emb_struct = emb_struct + self.alpha * emb_t1

                gate_func = torch.sigmoid(self.gate_func(torch.cat([emb_func, emb_asl], dim=-1)))
                emb_func = emb_func + gate_func * emb_asl

                gate_struct = torch.sigmoid(self.gate_struct(torch.cat([emb_struct, emb_t1], dim=-1)))
                emb_struct = emb_struct + gate_struct * emb_t1

            emb_func = self.linear(emb_func)
            emb_struct = self.linear(emb_struct)

            # === (7) 后续 cross-modal 融合 ===
            fused_func, attn_func = self.enhancer(enhanced_func, emb_func, emb_func)
            fused_struct, attn_struct = self.enhancer(enhanced_struct, emb_struct, emb_struct)

            final_embedding = torch.cat([fused_func, fused_struct], dim=-1)
            logits = self.classifier_bn(final_embedding)
            return logits
