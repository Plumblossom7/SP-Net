import torch
import numpy as np
import random
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dropout_adj

def make_graph_lists(A_funcs, X_funcs, A_structs, X_structs, labels, device):
    func_graphs, struct_graphs = [], []

    for Af, Xf, As, Xs, y in zip(A_funcs, X_funcs, A_structs, X_structs, labels):
        # --- Functional Graph ---
        rows_f, cols_f = torch.nonzero(Af, as_tuple=True)

        # edge_index_f
        edge_index_f = torch.stack([rows_f, cols_f], dim=0).to(torch.long).to(device)

        # edge_weight_f
        if isinstance(Af, torch.Tensor):
            edge_weight_f = Af[rows_f, cols_f].clone().detach().to(torch.float).to(device)
        else:
            edge_weight_f = torch.tensor(Af[rows_f, cols_f], dtype=torch.float, device=device)

        # x_f
        if isinstance(Xf, torch.Tensor):
            x_f = Xf.clone().detach().to(torch.float32).to(device)
        else:
            x_f = torch.tensor(Xf, dtype=torch.float32, device=device)

        y_f = torch.tensor([y], dtype=torch.long, device=device)

        data_f = Data(x=x_f, edge_index=edge_index_f, edge_attr=edge_weight_f, y=y_f)

        # --- Structural Graph ---
        rows_s, cols_s = torch.nonzero(As, as_tuple=True)
        edge_index_s = torch.stack([rows_s, cols_s], dim=0).to(torch.long).to(device)

        if isinstance(As, torch.Tensor):
            edge_weight_s = As[rows_s, cols_s].clone().detach().to(torch.float).to(device)
        else:
            edge_weight_s = torch.tensor(As[rows_s, cols_s], dtype=torch.float, device=device)

        if isinstance(Xs, torch.Tensor):
            x_s = Xs.clone().detach().to(torch.float32).to(device)
        else:
            x_s = torch.tensor(Xs, dtype=torch.float32, device=device)

        y_s = torch.tensor([y], dtype=torch.long, device=device)

        data_s = Data(x=x_s, edge_index=edge_index_s, edge_attr=edge_weight_s, y=y_s)

        func_graphs.append(data_f)
        struct_graphs.append(data_s)

    return func_graphs, struct_graphs

class IndexDataset(Dataset):
    # 保存索引以便数据加载时按索引组合
    def __init__(self, indices): self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.indices[idx]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def graph_augment(x, edge_index, edge_drop_prob=0.2, noise_scale=0.1):
    """
    图增强：随机丢边 + 特征加噪声
    Args:
        x: 节点特征 [num_nodes, num_features]
        edge_index: 图的边索引
        edge_drop_prob: 丢边比例
        noise_scale: 噪声强度（标准差）
    Returns:
        x_noisy: 加噪声后的节点特征
        edge_index_aug: 丢边后的边索引
    """
    # 1. Edge dropout
    edge_index_aug, _ = dropout_adj(edge_index, p=edge_drop_prob, force_undirected=True)

    # 2. Feature noise
    noise = torch.randn_like(x) * noise_scale
    x_noisy = x + noise

    return x_noisy, edge_index_aug

def calculate_performance(y_true, y_pred, y_scores):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    is_binary = len(labels) == 2
    average = 'binary' if is_binary else 'macro'

    # acc = (np.array(y_pred) == np.array(y_true)).mean()
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    # auc = roc_auc_score(y_true, y_scores)
    # cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # print("Confusion Matrix:\n", cm)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    try:
        # Binary evaluators in the 1106 pipeline return positive-class scores
        # as a 1-D list, while older callers may provide an [N, 2] matrix.
        binary_scores = y_scores[:, 1] if np.asarray(y_scores).ndim > 1 else y_scores
        auc = (roc_auc_score(y_true, binary_scores) if is_binary else
               roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro', labels=labels))
    except ValueError:
        auc = float('nan')  # 如果只有一个类，AUC无法计算
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion Matrix:\n", cm)

    return accuracy, recall, precision,  f1, auc, cm

def calculate_weighted_performance(y_true, y_pred, y_scores):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    import numpy as np

    accuracy = accuracy_score(y_true, y_pred)
    precision_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    try:
        if len(np.unique(y_true)) < 2:
            auc = float('nan')
        else:
            auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print("Class counts:", np.bincount(y_true))
    print("Confusion Matrix:\n", cm)
    print("Classification report:\n", classification_report(y_true, y_pred, target_names=['CN','MCI'], zero_division=0))

    return accuracy, recall_w, precision_w, f1_w, auc, cm
