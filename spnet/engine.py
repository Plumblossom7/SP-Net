import torch
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torch_geometric.data import DataLoader, Batch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def train_one_epoch(laux, lspe, func_graph, struct_graph, X_t1, X_asl, model, loader, optimizer, criterion, lcont, lothro, device, mode_with_aux=False, laux_cls=0.0):
    model.train()
    epoch_loss = 0
    epoch_cls_loss, epoch_cont_loss, epoch_orth_loss = 0, 0, 0
    y_true_all, y_pred_all = [], []

    for batch_idx in loader:
        bf = Batch.from_data_list([func_graph[i] for i in batch_idx]).to(device)
        bs = Batch.from_data_list([struct_graph[i] for i in batch_idx]).to(device)
        t1 = X_t1[batch_idx].to(device) if mode_with_aux else None
        asl = X_asl[batch_idx].to(device) if mode_with_aux else None

        out = model(bf.x, bf.edge_index, bf.edge_attr,
                    bs.x, bs.edge_index, bs.edge_attr,
                    bf.y, bf.batch, t1_input=t1, asl_input=asl, mode='train')
        cont_loss = out['contrastive_loss'] * lcont
        orth_loss = out['orthogonal_loss'] * lothro
        spe_loss = out['spe_cls_loss'] * lspe
        aux_loss = out['aux_loss'] * laux
        aux_cls_loss = out.get('aux_cls_loss', 0.0) * laux_cls
        cls_loss = criterion(out['logits'], bf.y)
        if mode_with_aux:
            aux_loss = out['aux_loss'] * laux
        else:
            aux_loss = 0


        loss = cls_loss + cont_loss + orth_loss + spe_loss + aux_loss + aux_cls_loss


        #loss = criterion(out['logits'], bf.y)

        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        #print("Grad norm:", total_norm)
        # 中等裁剪（常用）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_cls_loss += cls_loss.item()
        #epoch_cont_loss += cont_loss.item()
        #epoch_orth_loss += orth_loss.item()

        preds = out['logits'].argmax(dim=1)
        y_true_all.append(bf.y.cpu())
        y_pred_all.append(preds.cpu())

    # 拼接后计算 balanced accuracy
    y_true_all = torch.cat(y_true_all).numpy()
    y_pred_all = torch.cat(y_pred_all).numpy()
    train_bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)

    return epoch_loss / len(loader), train_bal_acc, epoch_cont_loss / len(loader), epoch_orth_loss / len(loader), epoch_cls_loss / len(loader)


def validate_one_epoch(func_graph, struct_graph, X_t1, X_asl, model, loader, criterion, device, mode_with_aux=False):

    # 验证集评估
    model.eval()
    correct, total = 0, 0
    val_loss_epoch = 0
    y_val_preds = []
    y_val_true = []
    with torch.no_grad():
        for batch_idx in loader:
            bf = Batch.from_data_list([func_graph[i] for i in batch_idx]).to(device)
            bs = Batch.from_data_list([struct_graph[i] for i in batch_idx]).to(device)
            t1 = X_t1[batch_idx].to(device) if mode_with_aux else None
            asl = X_asl[batch_idx].to(device) if mode_with_aux else None

            out = model(bf.x, bf.edge_index, bf.edge_attr,
                        bs.x, bs.edge_index, bs.edge_attr,
                        bf.y, bf.batch, t1_input=t1, asl_input=asl, mode='eval')

            preds = out.argmax(dim=1)
            val_loss = criterion(out, bf.y)
            val_loss_epoch += val_loss.item()
            correct += (preds == bf.y).sum().item()
            total   += bf.y.size(0)

            # ✅ append tensor 而不是 numpy
            y_val_preds.append(preds.cpu())
            y_val_true.append(bf.y.cpu())

    # ✅ 拼接 tensor
    y_val_preds = torch.cat(y_val_preds)
    y_val_true = torch.cat(y_val_true)

    # ✅ 转 numpy 计算指标
    y_val_preds_np = y_val_preds.numpy()
    y_val_true_np = y_val_true.numpy()


    val_bal_acc = balanced_accuracy_score(y_val_true_np, y_val_preds_np)
    #return val_loss_epoch / len(loader), correct / total, y_val_preds
    return val_loss_epoch / len(loader), val_bal_acc, y_val_preds, y_val_true



def evaluate_one_epoch(func_graph, struct_graph, X_t1, X_asl, model, loader, criterion, device, mode_with_aux=False):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    # zs_func_list, zs_struct_list = [], []
    # zp_func_list, zp_struct_list = [], []
    with torch.no_grad():
        for batch_idx in loader:
            bf = Batch.from_data_list([func_graph[i] for i in batch_idx]).to(device)
            bs = Batch.from_data_list([struct_graph[i] for i in batch_idx]).to(device)
            t1 = X_t1[batch_idx].to(device) if mode_with_aux else None
            asl = X_asl[batch_idx].to(device) if mode_with_aux else None

            out = model(bf.x, bf.edge_index, bf.edge_attr,
                        bs.x, bs.edge_index, bs.edge_attr,
                        bf.y, bf.batch, t1_input=t1, asl_input=asl, mode='eval')
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            y_true.extend(bf.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.detach().cpu().numpy())

    return y_true, y_pred, y_scores

class EarlyStopping:
    """
    训练早停机制：
    - 如果验证集损失在 patience 个 epoch 内没有显著下降，则触发 early_stop。
    - 通常用于防止过拟合，加快训练。

    参数:
    - patience: 最多容忍多少个 epoch 验证性能没有提升
    - delta: 最小改善值，小于该值认为未提升
    """
    def __init__(self, patience=20, delta=1e-4):
        self.patience = patience            # 容忍轮数
        self.counter = 0                    # 连续未提升轮数
        self.best_score = None              # 最佳得分（负 val_loss）
        self.early_stop = False             # 是否触发早停
        self.val_loss_min = np.inf          # 当前最小 val_loss
        self.delta = delta                  # 最小变化幅度

    def __call__(self, val_loss):
        score = -val_loss  # 越小越好，因此取负数越大越好

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # 达到 patience，触发早停

        else:
            self.best_score = score
            self.counter = 0  # 有提升，重置计数
            self.early_stop = False

def evaluate_val_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)
