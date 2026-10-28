"""Stable V3 SP-Net used by the final ADNI CN-versus-MCI experiment."""

import torch
import torch.nn.functional as F

from . import model as legacy_model


def bidirectional_contrastive_loss_stable(z_fc, z_struct, tau=0.07, lambda_param=0.5):
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    logits = F.normalize(z_fc, p=2, dim=1) @ F.normalize(z_struct, p=2, dim=1).T
    logits = logits / tau
    targets = torch.arange(logits.size(0), device=logits.device)
    return lambda_param * F.cross_entropy(logits, targets) + (1.0 - lambda_param) * F.cross_entropy(logits.T, targets)


def supervised_contrastive_loss(features, labels, temperature=0.1):
    if features.size(0) < 2:
        return torch.zeros((), device=features.device)
    features = F.normalize(features, p=2, dim=1)
    logits = (features @ features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    valid = ~identity
    positive = labels[:, None].eq(labels[None, :]) & valid
    log_denominator = torch.log((torch.exp(logits) * valid).sum(dim=1, keepdim=True).clamp_min(1e-12))
    log_probability = logits - log_denominator
    positive_count = positive.sum(dim=1)
    usable = positive_count > 0
    if not usable.any():
        return torch.zeros((), device=features.device)
    return -(log_probability * positive).sum(dim=1)[usable].div(positive_count[usable]).mean()


class MultiModalGNNStableV3(legacy_model.MultiModalGNN_new_new_new_new_new):
    """ROI-aligned BOLD/ASL and FA/T1 node fusion with stable InfoNCE."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_func.dropout.p = 0.3
        self.encoder_struct.dropout.p = 0.3
        for classifier in (self.classifier_bn, self.classifier_spe):
            for layer in classifier.modules():
                if isinstance(layer, torch.nn.Dropout) and layer.p == 0.5:
                    layer.p = 0.3
        # The public repository originally predates these two ROI-aligned
        # auxiliary projections.  Define them here so StableV3 remains
        # self-contained while retaining compatibility with that backbone.
        if not hasattr(self, "linear_asl"):
            self.linear_asl = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        if not hasattr(self, "linear_t1"):
            self.linear_t1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        with torch.no_grad():
            self.gate_func[-1].bias.fill_(2.1972246)  # sigmoid = 0.9
            self.gate_struct[-1].bias.fill_(2.1972246)

    def _node_embeddings(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch):
        if self.training and getattr(self, "dropedge_prob", 0.0) > 0:
            def drop_edges(edge_index, edge_weight):
                keep = torch.rand(edge_index.size(1), device=edge_index.device) >= self.dropedge_prob
                return edge_index[:, keep], edge_weight[keep]
            bf_edge_index, bf_edge_weight = drop_edges(bf_edge_index, bf_edge_weight)
            bs_edge_index, bs_edge_weight = drop_edges(bs_edge_index, bs_edge_weight)
        func_flat = self.encoder_func(bf_x, bf_edge_index, bf_edge_weight, batch)
        struct_flat = self.encoder_struct(bs_x, bs_edge_index, bs_edge_weight, batch)
        func_nodes, func_mask = legacy_model.to_dense_batch(func_flat, batch)
        struct_nodes, struct_mask = legacy_model.to_dense_batch(struct_flat, batch)
        if func_nodes.shape[:2] != struct_nodes.shape[:2] or not torch.equal(func_mask, struct_mask):
            raise ValueError("Functional and structural graphs must be ROI-aligned.")
        return func_nodes, struct_nodes, func_mask

    def _pool(self, values, batch):
        mode = getattr(self, "graph_pooling", "mean")
        mean = legacy_model.global_mean_pool(values, batch)
        if mode == "mean":
            return mean
        maximum = legacy_model.global_max_pool(values, batch)
        if mode == "max":
            return maximum
        if mode == "meanmax":
            return 0.5 * (mean + maximum)
        raise ValueError(f"Unsupported graph_pooling: {mode}")

    def forward(self, bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, labels, batch,
                t1_input, asl_input, mode="train"):
        func_nodes, struct_nodes, node_mask = self._node_embeddings(
            bf_x, bf_edge_index, bf_edge_weight, bs_x, bs_edge_index, bs_edge_weight, batch
        )
        aux_loss = torch.zeros((), device=func_nodes.device)
        if t1_input is not None and asl_input is not None:
            b, n, f = t1_input.shape
            if (b != func_nodes.size(0) or n != func_nodes.size(1) or asl_input.shape[:2] != (b, n)):
                raise ValueError("T1/ASL inputs must be aligned with graph nodes.")
            t1_nodes = self.t1_encoder(t1_input.reshape(b * n, f)).reshape(b, n, -1)
            asl_nodes = self.asl_encoder(asl_input.reshape(b * n, f)).reshape(b, n, -1)
            if t1_nodes.size(-1) != self.hidden_dim or asl_nodes.size(-1) != self.hidden_dim:
                raise ValueError("Node-level auxiliary fusion requires aux_dim == hidden.")
            asl_nodes = self.linear_asl(asl_nodes)
            t1_nodes = self.linear_t1(t1_nodes)
            if mode == "train":
                valid = node_mask.unsqueeze(-1).to(func_nodes.dtype)
                denominator = valid.sum(dim=1).clamp_min(1.0)
                func_pool = (func_nodes * valid).sum(dim=1) / denominator
                struct_pool = (struct_nodes * valid).sum(dim=1) / denominator
                asl_pool = (asl_nodes * valid).sum(dim=1) / denominator
                t1_pool = (t1_nodes * valid).sum(dim=1) / denominator
                aux_loss = (
                    legacy_model.correlation_penalty(func_pool.detach(), asl_pool)
                    + legacy_model.correlation_penalty(struct_pool.detach(), t1_pool)
                )
            bold_gate = torch.sigmoid(self.gate_func(torch.cat([func_nodes, asl_nodes], dim=-1)))
            fa_gate = torch.sigmoid(self.gate_struct(torch.cat([struct_nodes, t1_nodes], dim=-1)))
            func_nodes = (bold_gate * func_nodes + (1.0 - bold_gate) * asl_nodes).masked_fill(~node_mask.unsqueeze(-1), 0.0)
            struct_nodes = (fa_gate * struct_nodes + (1.0 - fa_gate) * t1_nodes).masked_fill(~node_mask.unsqueeze(-1), 0.0)

        func_flat, struct_flat = func_nodes[node_mask], struct_nodes[node_mask]
        zs_func_flat, zp_func_flat = self.head_func(func_flat)
        zs_struct_flat, zp_struct_flat = self.head_struct(struct_flat)
        zs_func, zp_func = self._pool(zs_func_flat, batch), self._pool(zp_func_flat, batch)
        zs_struct, zp_struct = self._pool(zs_struct_flat, batch), self._pool(zp_struct_flat, batch)
        emb_func, emb_struct = self._pool(func_flat, batch), self._pool(struct_flat, batch)

        logits_zp_func, logits_zp_struct = self.classifier_spe(zp_func), self.classifier_spe(zp_struct)
        probs_zp_func, probs_zp_struct = F.softmax(logits_zp_func, dim=1), F.softmax(logits_zp_struct, dim=1)
        gate_func = torch.sigmoid(self.gate_mlp_func(torch.cat([probs_zp_func.detach(), zp_func], dim=1)))
        gate_struct = torch.sigmoid(self.gate_mlp_struct(torch.cat([probs_zp_struct.detach(), zp_struct], dim=1)))
        fused_func = zs_func + 0.4 * gate_func * zp_func + self.linear(emb_func)
        fused_struct = zs_struct + 0.4 * gate_struct * zp_struct + self.linear(emb_struct)
        fused_features = torch.cat([fused_func, fused_struct], dim=-1)
        logits = self.classifier_bn(fused_features)
        if mode == "eval":
            return logits
        if mode != "train":
            raise ValueError(f"Unsupported mode: {mode}")
        return {
            "logits": logits,
            "spe_cls_loss": 0.5 * (F.cross_entropy(logits_zp_func, labels) + F.cross_entropy(logits_zp_struct, labels)),
            "contrastive_loss": bidirectional_contrastive_loss_stable(zs_func, zs_struct, self.tau, self.lambda_param),
            "orthogonal_loss": (
                legacy_model.orthogonal_loss(zs_func, zp_func)
                + legacy_model.orthogonal_loss(zs_struct, zp_struct)
                + legacy_model.orthogonal_loss(zp_func, zp_struct)
            ) / 3,
            "aux_loss": aux_loss,
            "supervised_contrastive_loss": supervised_contrastive_loss(
                fused_features, labels, getattr(self, "supcon_temperature", 0.1)
            ),
        }
