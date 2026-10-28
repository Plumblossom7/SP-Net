"""End-to-end training and evaluation helpers for StableV3."""

import weakref

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch_geometric.data import Batch


_model_epochs = weakref.WeakKeyDictionary()


def train_one_epoch(laux, lspe, func_graph, struct_graph, X_t1, X_asl, model, loader, optimizer, criterion,
                    lcont, lothro, device, mode_with_aux=True, warmup_epochs=15, lsupcon=0.0):
    model.train()
    epoch_number = _model_epochs.get(model, 0) + 1
    _model_epochs[model] = epoch_number
    warmup = min(1.0, epoch_number / float(warmup_epochs))
    total_loss = total_cls = total_cont = total_orth = 0.0
    y_true, y_pred = [], []
    for batch_idx in loader:
        bf = Batch.from_data_list([func_graph[i] for i in batch_idx]).to(device)
        bs = Batch.from_data_list([struct_graph[i] for i in batch_idx]).to(device)
        t1, asl = X_t1[batch_idx].to(device), X_asl[batch_idx].to(device)
        out = model(bf.x, bf.edge_index, bf.edge_attr, bs.x, bs.edge_index, bs.edge_attr, bf.y, bf.batch,
                    t1_input=t1, asl_input=asl, mode="train")
        cls_loss = criterion(out["logits"], bf.y)
        cont_loss = out["contrastive_loss"] * lcont
        orth_loss = out["orthogonal_loss"] * lothro
        spe_loss = out["spe_cls_loss"] * lspe
        aux_loss = out["aux_loss"] * laux if mode_with_aux else 0.0
        supcon_loss = out.get("supervised_contrastive_loss", 0.0) * lsupcon
        loss = cls_loss + warmup * (cont_loss + orth_loss + spe_loss + aux_loss + supcon_loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_cont += cont_loss.item()
        total_orth += orth_loss.item()
        y_true.append(bf.y.detach().cpu())
        y_pred.append(out["logits"].argmax(dim=1).detach().cpu())
    n_batches = len(loader)
    return (
        total_loss / n_batches,
        balanced_accuracy_score(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()),
        total_cont / n_batches,
        total_orth / n_batches,
        total_cls / n_batches,
    )


def evaluate(func_graph, struct_graph, X_t1, X_asl, model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for batch_idx in loader:
            bf = Batch.from_data_list([func_graph[i] for i in batch_idx]).to(device)
            bs = Batch.from_data_list([struct_graph[i] for i in batch_idx]).to(device)
            logits = model(bf.x, bf.edge_index, bf.edge_attr, bs.x, bs.edge_index, bs.edge_attr, bf.y, bf.batch,
                           t1_input=X_t1[batch_idx].to(device), asl_input=X_asl[batch_idx].to(device), mode="eval")
            total_loss += criterion(logits, bf.y).item()
            probabilities = torch.softmax(logits, dim=1)
            y_true.extend(bf.y.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
            y_score.extend(probabilities[:, 1].cpu().numpy())
    return total_loss / len(loader), np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)


# Kept separate for a clear, lightweight public interface.
import numpy as np


def validate_one_epoch(*args, **kwargs):
    loss, y_true, y_pred, _ = evaluate(*args, **kwargs)
    return loss, accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred), y_pred


def evaluate_one_epoch(*args, **kwargs):
    _, y_true, y_pred, y_score = evaluate(*args, **kwargs)
    return y_true, y_pred, y_score
