"""Train SP-Net on the ID-aligned four-modality ADNI CN-versus-MCI cohort."""

import os
import random
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time
import torch.nn as nn
import seaborn as sns

from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spnet.data import ADNIFourModalDataset

from spnet.model import MultiModalGNN, MultiModalGNN_new_new_new_new_new
from spnet.graph import make_graph_lists, set_seed, IndexDataset, calculate_performance
from spnet.engine import train_one_epoch, validate_one_epoch, evaluate_one_epoch, EarlyStopping
from spnet.plots import plot_train_val, plot_train_val_all
from spnet.confusion import plot_confuse_matrix
from spnet.attribution import node_importance, normalize_attributions, node_importance_node_level

from sklearn.utils.class_weight import compute_class_weight
from captum.attr import IntegratedGradients

import os
from datetime import datetime
import itertools
from itertools import product

# 脚本开始时设置工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))



def build_subject_prediction_dataframe(fold, split, original_indices, y_true, y_pred, probabilities,
                                       classes, subject_id_by_original_index):
    """Create an ID-resolved prediction table for one test split."""
    original_indices = np.asarray(original_indices, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = np.column_stack((1.0 - probabilities, probabilities))
    if not (len(original_indices) == len(y_true) == len(y_pred) == len(probabilities)):
        raise ValueError(f'{split} fold {fold}: prediction arrays and subject indices have inconsistent lengths.')

    rows = []
    for index, true_label, pred_label, probs in zip(original_indices, y_true, y_pred, probabilities):
        row = {
            'fold': fold,
            'split': split,
            'original_index': int(index),
            'subject_id': subject_id_by_original_index.get(int(index), f'row_{int(index):03d}'),
            'true_label': int(true_label),
            'true_diagnosis': classes[int(true_label)],
            'predicted_label': int(pred_label),
            'predicted_diagnosis': classes[int(pred_label)],
            'predicted_probability': float(probs[int(pred_label)]),
        }
        row.update({f'probability_{class_name}': float(probs[class_index])
                    for class_index, class_name in enumerate(classes)})
        rows.append(row)
    return pd.DataFrame(rows)


def train_multimodal_grid_cv(hparams, classes):

    root_folder = hparams['folder_name']
    os.makedirs(root_folder, exist_ok=True)
    os.makedirs(f'{root_folder}/conf', exist_ok=True)
    os.makedirs(f'{root_folder}/node_importance', exist_ok=True)
    #os.makedirs(f'{root_folder}/x_joint', exist_ok=True)
    # os.makedirs(f'{root_folder}/zp_func', exist_ok=True)
    # os.makedirs(f'{root_folder}/zp_struct', exist_ok=True)

    # Source-array and graph validations run before CV; avoid the large
    # runtime cost of autograd anomaly tracing during the full ADNI training.
    torch.autograd.set_detect_anomaly(False)
    #os.makedirs("curve", exist_ok=True)

    # 添加TensorBoard writer
    #writer = SummaryWriter(f'runs/multimodal_gnn_{time.strftime("%Y%m%d-%H%M%S")}')

    set_seed(hparams['random_seed'])

    # 加载数据
    # ADNI source loading is ID-aligned to the exported CN/MCI participant table.
    ds = ADNIFourModalDataset(
        mat_path=hparams['mat_path'],
        asl_feature_path=hparams['asl_feature_path'],
        subject_table_path=hparams['subject_table_path'],
        random_seed=hparams['random_seed'],
        keep_ratio=hparams['keep_ratio'],
        dti_feature_mode=hparams['dti_feature_mode'],
        fc_mode=hparams['fc_mode'],
        fn_dir=hparams['fn_dir'],
        split_mode='five_fold',
    )
    '''
    dataset_cls = ToyDatasetDualGraphAgeResidualized if hparams.get('age_residualize', False) else ToyDatasetDualGraph
    dataset_kwargs = {}
    if hparams.get('age_residualize', False):
        dataset_kwargs['age_table_path'] = hparams['age_table_path']
    ds = dataset_cls(
        graymatter_path=hparams['t1_path'],
        cbf_path=hparams['cbf_path'],
        bold_path=hparams['bold_path'],
        fa_path=hparams['fa_path'],
        fn_dir=hparams['fn_dir'],
        label_path=hparams['label_path'],
        binary_class=hparams.get('binary_class', False),
        task_labels=hparams.get('task_labels'),
        **dataset_kwargs,
        random_seed=hparams['random_seed'],
        keep_ratio=hparams['keep_ratio'],  # 新增参数：保留边的比例
        threshold=hparams['threshold'],  # 新增参数：结构连接矩阵的阈值
    )

    if hparams.get('age_residualize', False):
        coeff_summary = {
            fold: {key: value for key, value in records.items() if key not in ('func', 'struct', 'asl', 't1')}
            for fold, records in ds.age_residualization_coefficients.items()
        }
        with open(os.path.join(root_folder, 'age_residualization_fold_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(coeff_summary, f, ensure_ascii=False, indent=2)

    '''
    def _jsonable(value):
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        return value

    # Save the complete effective configuration, including non-grid options.
    run_config = _jsonable(dict(hparams))
    run_config.update({
        'classes': list(classes),
        'num_subjects': int(len(ds.y)),
        'fold_test_sizes': [int(len(ds(cv, tensor=True)['y'][2])) for cv in range(1, 6)],
        'scheduler': {
            'type': 'ReduceLROnPlateau',
            'mode': 'min',
            'factor': hparams['scheduler_factor'],
            'patience': hparams['scheduler_patience'],
            'threshold': hparams['scheduler_threshold'],
            'min_lr': hparams['scheduler_min_lr'],
        },
    })
    print('\n===== Effective run configuration =====')
    print(json.dumps(run_config, ensure_ascii=False, indent=2, sort_keys=True))
    with open(os.path.join(root_folder, 'run_parameters.json'), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    pd.DataFrame([run_config]).to_csv(
        os.path.join(root_folder, 'run_parameters.csv'), index=False
    )

    subject_id_by_original_index = {
        int(original_index): str(subject_id)
        for original_index, subject_id in zip(
            getattr(ds, 'original_indices', []), getattr(ds, 'subject_ids', [])
        )
    }

    test_accs = []
    test_recalls = []
    test_precisions = []
    test_f1s = []
    test_aucs = []
    all_y_true = []
    all_y_pred = []
    all_test_predictions = []
    best_params_cv = []
    fold_weight_records = []
    fold_selection_records = []
    all_fold_attributions_func, all_fold_attributions_struct, all_fold_attributions_mean = [], [], []  # 存储5折的节点贡献

    for cv in tqdm(range(1, 6), desc='Cross Validation'):
        print(f"\n===== Grid Search on CV {cv} =====")

        sub_folder = os.path.join(root_folder, f"fold_{cv}")
        save_dir_1 = os.path.join(sub_folder, f"stage_1")
        save_dir_2 = os.path.join(sub_folder, f"stage_2")
        save_dir_3 = os.path.join(sub_folder, f"stage_3")
        os.makedirs(sub_folder, exist_ok=True)
        os.makedirs(save_dir_1, exist_ok=True)
        os.makedirs(save_dir_2, exist_ok=True)
        os.makedirs(save_dir_3, exist_ok=True)

        data = ds(cv, tensor=True)

        # 拆分数据
        Af_train, Af_val, Af_test = data['A_func']
        Xf_train, Xf_val, Xf_test = data['X_func']
        As_train, As_val, As_test = data['A_struct']
        Xs_train, Xs_val, Xs_test = data['X_struct']
        y_train,  y_val,  y_test  = data['y']
        original_train_indices, original_val_indices, original_test_indices = data['indices']
        Xa_train, Xa_val, Xa_test = data['X_asl']
        Xt_train, Xt_val, Xt_test = data['X_t1']
        print(
            f'Fold {cv} split sizes: train={len(y_train)}, val={len(y_val)}, '
            f'test={len(y_test)}; train label counts={np.bincount(y_train).tolist()}'
        )

        device = hparams['device']

        #还有更快的数据加载方式：先构图，再对图数据划分数据集，这样省下了每次运行构图的时间，后面再改
        fg_train, sg_train = make_graph_lists(Af_train, Xf_train, As_train, Xs_train, y_train, device)
        fg_val, sg_val = make_graph_lists(Af_val, Xf_val, As_val, Xs_val, y_val, device)
        fg_test, sg_test = make_graph_lists(Af_test, Xf_test, As_test, Xs_test, y_test, device)

        # 构建索引数据集
        idx_train = list(range(len(fg_train)))
        idx_val = list(range(len(fg_val)))
        idx_test = list(range(len(fg_test)))
        loader_train = DataLoader(IndexDataset(idx_train), batch_size=hparams['batch_size'], shuffle=True)
        loader_val = DataLoader(IndexDataset(idx_val), batch_size=hparams['batch_size'], shuffle=False)
        loader_test = DataLoader(IndexDataset(idx_test), batch_size=hparams['batch_size'], shuffle=False)


        # 假设 num_classes = 2
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(hparams['num_classes']),
            y=y_train  # 你的标签
        )
        fold_weight_records.append({
            'fold': cv,
            **{f'class_{class_id}_train_count': int((np.asarray(y_train) == class_id).sum())
               for class_id in range(hparams['num_classes'])},
            **{f'class_{class_id}_balanced_weight': float(class_weights[class_id])
               for class_id in range(hparams['num_classes'])},
        })
        # Keep automatic balanced weights, with an optional task-specific
        # down-scaling for the positive/minority class.
        mode = hparams['mode']   # "normal" "weighted_loss"
        norm = hparams['norm']

        best_val_acc, best_params, best_model_cv= 0, 0, None


        param_grid = {
            'lr_1': hparams['lr_1'],
            'lr_2': hparams['lr_2'],
            'lr_3': hparams['lr_3'],
            'reg': hparams['reg'],
            'lcont': hparams['lcont'],
            'lothro': hparams['lothro'],
            'aux_dim': hparams['aux_dim'],
            'shared_dim': hparams['shared_dim'],
            'tau': hparams['tau'],
            'lambda_param': hparams['lambda_param'],
            'heads': hparams['heads'],
            'hidden': hparams['hidden'],
            'cluster': hparams['cluster'],
            'lspe': hparams['lspe'],
            'laux': hparams['laux'],
            'laux_cls': hparams['laux_cls'],
            'minority_weight_scale': hparams.get('minority_weight_scale', [1.0]),
        }
        # Dimension-only tuning can preserve the saved best objective settings
        # independently for each outer fold.
        fold_fixed = hparams.get('fold_fixed_params', {}).get(cv, {})
        for fixed_name, fixed_value in fold_fixed.items():
            param_grid[fixed_name] = [fixed_value]

        # keys, values = zip(*param_grid.items())
        # for param_tuple in itertools.product(*values):
        #     params = dict(zip(keys, param_tuple))
        # n_samples = 2  # 想要随机搜索多少组参数
        # for _ in range(n_samples): #随机搜索
        #     params = {k: random.choice(v) for k, v in param_grid.items()}
        #     print(params)
        # 生成所有组合
        keys = list(param_grid.keys())
        all_combinations = list(product(*param_grid.values()))
        random.shuffle(all_combinations)  # 打乱

        # n_samples = 2  # 想要随机搜索多少组参数
        n_samples = min(len(all_combinations), hparams.get('max_grid_samples', len(all_combinations)))
        for i in range(n_samples):
            combo = all_combinations[i]
            params = dict(zip(keys, combo))
            if hparams.get('tie_aux_dim_to_hidden', False):
                params['aux_dim'] = params['hidden']
            print(params)

            # Rebuild the weighted objective for every candidate.  For MCI-AD,
            # the raw balanced weight for AD can be too aggressive because the
            # class ratio is about 70:30; searching this scale also searches the
            # operating point learned by the classifier without touching test data.
            candidate_class_weights = class_weights.copy() if 'class_weights' in locals() else compute_class_weight(
                class_weight="balanced",
                classes=np.arange(hparams['num_classes']),
                y=y_train
            )
            minority_weight_scale = params.get('minority_weight_scale', 1.0)
            if hparams['num_classes'] == 2 and minority_weight_scale != 1.0:
                candidate_class_weights[1] *= minority_weight_scale
            candidate_class_weights = torch.tensor(candidate_class_weights, dtype=torch.float32).to(device)
            criterion_weighted = nn.CrossEntropyLoss(weight=candidate_class_weights)

            model = MultiModalGNN_new_new_new_new_new(
                in_dim1=fg_train[0].x.shape[1],
                in_dim2=sg_train[0].x.shape[1],
                aux_dim=params['aux_dim'],
                shared_dim=params['shared_dim'],
                private_dim=params['shared_dim'] ,
                num_classes=hparams['num_classes'],
                tau=params['tau'],
                lambda_param=params['lambda_param'],
                hidden=params['hidden'],
                norm=hparams['norm'][0],
            ).to(device)
            if hparams.get('dropout') is not None:
                for module in model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = hparams['dropout']

            stopper_1 = EarlyStopping(patience=hparams['patience_1'], delta=hparams['delta'])
            stopper_2 = EarlyStopping(patience=hparams['patience_2'], delta=hparams['delta'])
            stopper_3 = EarlyStopping(patience=hparams['patience_3'], delta=hparams['delta'])

            crit = nn.CrossEntropyLoss()
            torch.cuda.empty_cache()

            # 初始化准确率和损失记录列表
            train_losses_epoch = []
            val_losses_epoch = []
            val_accs_epoch = []
            train_accs_epoch = []  # 添加训练准确率记录列表



            desc1=f"[CV{cv} lr={params.get('lr_1')} reg={params.get('reg')} lcont={params.get('lcont')} lothro={params.get('lothro')} cluster={params.get('cluster')} shared_dim={params.get('shared_dim')} tau={params.get('tau')} lambda_param={params.get('lambda_param')} hidden={params.get('hidden')}]"
            desc2=f"[CV{cv} lr={params.get('lr_2')} reg={params.get('reg')} lcont={params.get('lcont')} lothro={params.get('lothro')} cluster={params.get('cluster')} shared_dim={params.get('shared_dim')} tau={params.get('tau')} lambda_param={params.get('lambda_param')} hidden={params.get('hidden')}]"
            desc3=f"[CV{cv} lr={params.get('lr_3')} reg={params.get('reg')} lcont={params.get('lcont')} lothro={params.get('lothro')} cluster={params.get('cluster')} shared_dim={params.get('shared_dim')} tau={params.get('tau')} lambda_param={params.get('lambda_param')} hidden={params.get('hidden')}]"
            pbar_1 = tqdm(range(hparams['epochs_1'][0]), desc=desc1)  # 只用第一个epoch数值
            pbar_2 = tqdm(range(hparams['epochs_2'][0]), desc=desc2)  # 只用第一个epoch数值
            pbar_3 = tqdm(range(hparams['epochs_3'][0]), desc=desc3)  # 只用第一个epoch数值

            #训练+验证
            #第一阶段训练
            if not hparams.get('run_stage3', True):
                pbar_3 = tqdm([], desc='Stage 3 disabled')
            print("Stage 1: training main model (DTI + fMRI)")
            model.freeze_t1_asl()  # 冻结T1+ASL，只训练主模态
            opt_1 = torch.optim.Adam(model.parameters(), lr=params['lr_1'], weight_decay=params['reg'])
            scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_1, mode='max', factor=hparams['scheduler_factor'], patience=hparams['scheduler_patience'],
                threshold=hparams['scheduler_threshold'], min_lr=hparams['scheduler_min_lr']
            )
            best_model_epoch_1 = None
            best_val_f1_epoch_1 = 0
            train_losses_epoch_1 = []
            val_losses_epoch_1 = []
            val_accs_epoch_1 = []
            train_accs_epoch_1 = []  # 添加训练准确率记录列表
            val_f1s_epoch_1 = []
            for epoch in pbar_1:
                #训练集训练
                model.train()
                epoch_loss = 0
                correct, total = 0, 0

                crit_train = crit
                if mode == "weighted_loss":
                    crit_train = criterion_weighted

                loss, corr, _, _, _ = train_one_epoch(params['laux'], params['lspe'], func_graph=fg_train, struct_graph=sg_train, X_t1=Xt_train, X_asl=Xa_train, model=model, loader=loader_train, optimizer=opt_1, criterion=crit_train, lcont=params['lcont'], lothro=params['lothro'], device=device, mode_with_aux=False, laux_cls=0.0)

                train_losses_epoch.append(loss)
                # epoch结束后计算准确率
                train_accs_epoch.append(100.0 * corr)  # 记录平均训练准确率

                train_losses_epoch_1.append(loss)
                # epoch结束后计算准确率
                train_accs_epoch_1.append(100.0 * corr)  # 记录平均训练准确率


                # 验证集评估
                val_loss, val_corr, y_val_preds, y_val_true = validate_one_epoch(func_graph=fg_val, struct_graph=sg_val, X_t1=Xt_val, X_asl=Xa_val, model=model, loader=loader_val, criterion=crit, device=device, mode_with_aux=False)
                val_f1 = f1_score(y_val_true.numpy(), y_val_preds.numpy(), average='macro', zero_division=0)
                val_bal_acc = val_corr
                val_corr = val_f1

                val_accs_epoch.append(100.0 * val_bal_acc)
                val_losses_epoch.append(val_loss)
                scheduler_1.step(val_f1)

                val_accs_epoch_1.append(100.0 * val_bal_acc)
                val_losses_epoch_1.append(val_loss)
                val_f1s_epoch_1.append(val_f1)


                if val_f1 >= best_val_f1_epoch_1:
                    best_val_f1_epoch_1 = val_f1
                    best_model_epoch_1 = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(save_dir_1, f"loss_lr{params['lr_1']}_reg{params['reg']}_lcont{params['lcont']}_lothro{params['lothro']}_shared_dim{params['shared_dim']}_tau{params['tau']}_lambda{params['lambda_param']}_hidden{params['hidden']}_lspe{params['lspe']}_best_stage1.pth"))
                if not hparams.get('disable_early_stopping', False):
                    stopper_1(1 - val_corr)  # 1 - val_acc represents error rate
                    if stopper_1.early_stop:
                        break
                #训练过程中直接监视指标
                #pbar.set_postfix({'TrainLoss': f"{train_losses_epoch[-1]:.4f}", 'ValLoss': f"{val_losses_epoch[-1]:.4f}", 'ValAcc': f"{val_acc_epoch:.4f}"})
            plot_train_val(train_losses_epoch_1, val_losses_epoch_1, train_accs_epoch_1, val_accs_epoch_1, save_dir_1, params['lr_1'], params['reg'], params['lcont'], params['lothro'], params['shared_dim'], params['tau'], params['lambda_param'], params['hidden'], params['lspe'], params['laux'])

            # ---------- Stage 2 ----------
            print("Stage 2: end-to-end joint training of main + T1/ASL modalities")
            # 尝试调用模型自带 freeze，如果没有则通过命名冻结
            # Jointly fine-tune all modalities after the main-modality warmup.
            # T1/ASL must be able to improve the shared classifier rather than
            # adapting while the main classification path remains frozen.
            model.unfreeze_all()
            if best_model_epoch_1 is not None:
                model.load_state_dict(best_model_epoch_1)


            # 只优化 requires_grad=True 的参数
            opt_2 = torch.optim.Adam(model.parameters(), lr=params['lr_2'], weight_decay=params['reg'])
            scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_2, mode='max', factor=hparams['scheduler_factor'], patience=hparams['scheduler_patience'],
                threshold=hparams['scheduler_threshold'], min_lr=hparams['scheduler_min_lr']
            )
            best_model_epoch_2 = None
            best_val_f1_epoch_2 = 0
            train_losses_epoch_2 = []
            val_losses_epoch_2 = []
            val_accs_epoch_2 = []
            train_accs_epoch_2 = []  # 添加训练准确率记录列表
            val_f1s_epoch_2 = []
            for epoch in pbar_2:
                #训练集训练
                model.train()
                epoch_loss = 0
                correct, total = 0, 0

                crit_train = crit
                if mode == "weighted_loss":
                    crit_train = criterion_weighted

                loss, corr, _, _, _ = train_one_epoch(params['laux'], params['lspe'], func_graph=fg_train, struct_graph=sg_train, X_t1=Xt_train, X_asl=Xa_train, model=model, loader=loader_train, optimizer=opt_2, criterion=crit_train, lcont=params['lcont'], lothro=params['lothro'], device=device, mode_with_aux=True, laux_cls=params['laux_cls'])

                train_losses_epoch.append(loss)
                # epoch结束后计算准确率
                train_accs_epoch.append(100.0 * corr)  # 记录平均训练准确率

                train_losses_epoch_2.append(loss)
                # epoch结束后计算准确率
                train_accs_epoch_2.append(100.0 * corr)  # 记录平均训练准确率


                # 验证集评估
                val_loss, val_corr, y_val_preds, y_val_true = validate_one_epoch(func_graph=fg_val, struct_graph=sg_val, X_t1=Xt_val, X_asl=Xa_val, model=model, loader=loader_val, criterion=crit, device=device, mode_with_aux=True)
                val_f1 = f1_score(y_val_true.numpy(), y_val_preds.numpy(), average='macro', zero_division=0)
                val_bal_acc = val_corr
                val_corr = val_f1

                val_accs_epoch.append(100.0 * val_bal_acc)
                val_losses_epoch.append(val_loss)

                val_accs_epoch_2.append(100.0 * val_bal_acc)
                val_losses_epoch_2.append(val_loss)
                val_f1s_epoch_2.append(val_f1)
                scheduler_2.step(val_f1)


                if val_f1 >= best_val_f1_epoch_2:
                    best_val_f1_epoch_2 = val_f1
                    best_model_epoch_2 = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(save_dir_2, f"loss_lr{params['lr_2']}_reg{params['reg']}_lcont{params['lcont']}_lothro{params['lothro']}_shared_dim{params['shared_dim']}_tau{params['tau']}_lambda{params['lambda_param']}_hidden{params['hidden']}_lspe{params['lspe']}_best_stage2.pth"))
                if not hparams.get('disable_early_stopping', False):
                    stopper_2(1 - val_corr)  # 1 - val_acc represents error rate
                    if stopper_2.early_stop:
                        break
                #训练过程中直接监视指标
                #pbar.set_postfix({'TrainLoss': f"{train_losses_epoch[-1]:.4f}", 'ValLoss': f"{val_losses_epoch[-1]:.4f}", 'ValAcc': f"{val_acc_epoch:.4f}"})
            plot_train_val(train_losses_epoch_2, val_losses_epoch_2, train_accs_epoch_2, val_accs_epoch_2, save_dir_2, params['lr_2'], params['reg'], params['lcont'], params['lothro'], params['shared_dim'], params['tau'], params['lambda_param'], params['hidden'], params['lspe'], params['laux'])

            # Start the final low-LR fine-tuning from the best joint checkpoint,
            # not from the last (possibly degraded) Stage-2 epoch.
            if best_model_epoch_2 is not None:
                model.load_state_dict(best_model_epoch_2)


            # ---------- Stage 3: optional compatibility slot ----------
            if not hparams.get('run_stage3', True):
                print("Two-stage mode: Stage 3 disabled")
            else:
                print("Stage 3: unfreeze and fine-tune full model (optional)")
            model.unfreeze_all()  # or implement per-module unfreeze
            opt_3 = torch.optim.Adam(model.parameters(), lr=params['lr_3'], weight_decay=params['reg'])
            scheduler_3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt_3, mode='max', factor=hparams['scheduler_factor'], patience=hparams['scheduler_patience'],
                threshold=hparams['scheduler_threshold'], min_lr=hparams['scheduler_min_lr']
            )
            best_model_epoch_3 = None
            best_val_f1_epoch_3 = 0
            train_losses_epoch_3 = []
            val_losses_epoch_3 = []
            val_accs_epoch_3 = []
            train_accs_epoch_3 = []  # 添加训练准确率记录列表
            val_f1s_epoch_3 = []
            for epoch in pbar_3:
                #训练集训练
                model.train()
                epoch_loss = 0
                correct, total = 0, 0

                crit_train = crit
                if mode == "weighted_loss":
                    crit_train = criterion_weighted

                loss, corr, _, _, _ = train_one_epoch(params['laux'], params['lspe'], func_graph=fg_train, struct_graph=sg_train, X_t1=Xt_train, X_asl=Xa_train, model=model, loader=loader_train, optimizer=opt_3, criterion=crit_train, lcont=params['lcont'], lothro=params['lothro'], device=device, mode_with_aux=True, laux_cls=params['laux_cls'])

                train_losses_epoch.append(loss)
                # epoch结束后计算准确率
                train_accs_epoch.append(100.0 * corr)  # 记录平均训练准确率

                train_losses_epoch_3.append(loss)
                # epoch结束后计算准确率
                train_accs_epoch_3.append(100.0 * corr)  # 记录平均训练准确率


                # 验证集评估
                val_loss, val_corr, y_val_preds, y_val_true = validate_one_epoch(func_graph=fg_val, struct_graph=sg_val, X_t1=Xt_val, X_asl=Xa_val, model=model, loader=loader_val, criterion=crit, device=device, mode_with_aux=True)
                val_f1 = f1_score(y_val_true.numpy(), y_val_preds.numpy(), average='macro', zero_division=0)
                val_bal_acc = val_corr
                val_corr = val_f1

                val_accs_epoch.append(100.0 * val_bal_acc)
                val_losses_epoch.append(val_loss)

                val_accs_epoch_3.append(100.0 * val_bal_acc)
                val_losses_epoch_3.append(val_loss)
                val_f1s_epoch_3.append(val_f1)
                scheduler_3.step(val_f1)


                if val_f1 >= best_val_f1_epoch_3:
                    best_val_f1_epoch_3 = val_f1
                    best_model_epoch_3 = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(save_dir_3, f"loss_lr{params['lr_3']}_reg{params['reg']}_lcont{params['lcont']}_lothro{params['lothro']}_shared_dim{params['shared_dim']}_tau{params['tau']}_lambda{params['lambda_param']}_hidden{params['hidden']}_lspe{params['lspe']}_best_stage3.pth"))
                if not hparams.get('disable_early_stopping', False):
                    stopper_3(1 - val_corr)  # 1 - val_acc represents error rate
                    if stopper_3.early_stop:
                        break
                #训练过程中直接监视指标
                #pbar.set_postfix({'TrainLoss': f"{train_losses_epoch[-1]:.4f}", 'ValLoss': f"{val_losses_epoch[-1]:.4f}", 'ValAcc': f"{val_acc_epoch:.4f}"})
            if not hparams.get('run_stage3', True):
                best_model_epoch_3 = best_model_epoch_2
                best_val_f1_epoch_3 = best_val_f1_epoch_2
            else:
                plot_train_val(train_losses_epoch_3, val_losses_epoch_3, train_accs_epoch_3, val_accs_epoch_3, save_dir_3, params['lr_3'], params['reg'], params['lcont'], params['lothro'], params['shared_dim'], params['tau'], params['lambda_param'], params['hidden'], params['lspe'], params['laux'])




            # # 保存该参数组合的学习曲线{lcont} lothro={lothro} shared_dim={shared_dim} tau={tau} lambda_param={lambda_param} heads={heads} hidden={hidden} num_layers={num_layers}')
            plot_train_val_all(train_losses_epoch, val_losses_epoch, train_accs_epoch, val_accs_epoch, sub_folder, params['lr_1'], params['lr_2'], params['lr_3'], params['reg'], params['lcont'], params['lothro'], params['shared_dim'], params['tau'], params['lambda_param'], params['hidden'], params['lspe'], params['laux'])
            # Preserve raw epoch-level values alongside PNGs so overfitting
            # curves can be inspected or averaged across folds later.
            histories = []
            for stage, train_loss, val_loss, train_acc, val_acc, val_f1s in (
                ('stage_1', train_losses_epoch_1, val_losses_epoch_1, train_accs_epoch_1, val_accs_epoch_1, val_f1s_epoch_1),
                ('stage_2', train_losses_epoch_2, val_losses_epoch_2, train_accs_epoch_2, val_accs_epoch_2, val_f1s_epoch_2),
                ('stage_3', train_losses_epoch_3, val_losses_epoch_3, train_accs_epoch_3, val_accs_epoch_3, val_f1s_epoch_3),
            ):
                histories.extend({
                    'stage': stage,
                    'epoch_in_stage': epoch + 1,
                    'train_loss': train_loss[epoch],
                    'validation_loss': val_loss[epoch],
                    'train_accuracy': train_acc[epoch],
                    'validation_accuracy': val_acc[epoch],
                    'validation_macro_f1': val_f1s[epoch],
                } for epoch in range(len(train_loss)))
            pd.DataFrame(histories).to_csv(
                os.path.join(sub_folder, 'learning_curve_history.csv'), index=False
            )

            #比较每一组参数的验证集准确率，保存最好的参数
            if  best_val_f1_epoch_3 >= best_val_acc:
                best_val_acc = best_val_f1_epoch_3
                best_params = {'lr': params['lr_3'], 'lr_1': params['lr_1'], 'lr_2': params['lr_2'], 'reg': params['reg'], 'lcont': params['lcont'], 'lothro': params['lothro'], 'shared_dim': params['shared_dim'], 'aux_dim': params['aux_dim'], 'tau': params['tau'], 'lambda_param': params['lambda_param'], 'hidden': params['hidden'], 'lspe': params['lspe'], 'laux': params['laux'], 'laux_cls': params['laux_cls'], 'minority_weight_scale': params.get('minority_weight_scale', 1.0)}
                best_model_cv = best_model_epoch_3
                #best_model_cv = copy.deepcopy(model.state_dict())
                print('best_val_macro_f1_epoch', best_val_f1_epoch_3)
                print('\n')

        # 该折的内部网格调参结束了，评估测试集的表现
        # 测试集评估
        # 将保存的参数加载到模型中
        model = MultiModalGNN_new_new_new_new_new(
                in_dim1=fg_train[0].x.shape[1],
                in_dim2=sg_train[0].x.shape[1],
                aux_dim=best_params['aux_dim'],
                shared_dim=best_params['shared_dim'],
                private_dim=best_params['shared_dim'] ,
                num_classes=hparams['num_classes'],
                tau=best_params['tau'],
                lambda_param=best_params['lambda_param'],
                hidden=best_params['hidden'],
                norm=hparams['norm'][0],
            ).to(device)
        if hparams.get('dropout') is not None:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = hparams['dropout']
        model.load_state_dict(best_model_cv)

        y_true, y_pred, y_scores = evaluate_one_epoch(func_graph=fg_test, struct_graph=sg_test, X_t1=Xt_test, X_asl=Xa_test, model=model, loader=loader_test, criterion=criterion_weighted, device=device, mode_with_aux=True)
        test_predictions = build_subject_prediction_dataframe(
            cv, 'test', original_test_indices, y_true, y_pred, y_scores,
            classes, subject_id_by_original_index
        )
        test_predictions.to_csv(os.path.join(sub_folder, 'test_subject_predictions.csv'), index=False)
        all_test_predictions.append(test_predictions)

        acc, recall, precision, f1, auc, cm = calculate_performance(y_true, y_pred, y_scores)

        test_accs.append(acc)
        test_recalls.append(recall)
        test_precisions.append(precision)
        test_f1s.append(f1)
        test_aucs.append(auc)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        best_params_cv.append(best_params)
        fold_selection_records.append({
            'fold': cv,
            'best_validation_macro_f1': float(best_val_acc),
            **best_params,
        })
        print('y_test_preds', y_pred)


        plot_confuse_matrix(cm, classes, root_folder, cv)


        '''captum节点可解释性'''
        if hparams.get('run_node_importance', True):
            node_importance_func_norm, node_importance_struct_norm, node_importance_mean_norm = node_importance_node_level(
                fg_test, sg_test, Xt_test, Xa_test, model, loader_test, device
            )
            all_fold_attributions_func.append(node_importance_func_norm.cpu().numpy())
            all_fold_attributions_struct.append(node_importance_struct_norm.cpu().numpy())
            all_fold_attributions_mean.append(node_importance_mean_norm.cpu().numpy())

    avg_acc, std_acc = np.mean(test_accs), np.std(test_accs)
    print(f"\nFinal Mean Test Acc: {avg_acc:.4f} ± {std_acc:.4f}")
    df = pd.DataFrame({
        'Fold': list(range(1, len(test_accs) + 1)),  # 1到5的fold编号
        'Test_Acc': test_accs,
        'Test_Recall': test_recalls,
        'Test_Precision': test_precisions,
        'Test_F1': test_f1s,
        'Test_AUC': test_aucs,
        'Y_true': all_y_true,
        'Y_pred': all_y_pred
    })
    # 先算平均 & 标准差，并保留三位小数
    mean_row = df[['Test_Acc', 'Test_Recall', 'Test_Precision', 'Test_F1', 'Test_AUC']].mean().round(4)
    std_row  = df[['Test_Acc', 'Test_Recall', 'Test_Precision', 'Test_F1', 'Test_AUC']].std().round(4)

    # 拼接成 DataFrame
    summary = pd.DataFrame({
        'Fold': ['Mean', 'Std'],
        'Test_Acc': [mean_row['Test_Acc'], std_row['Test_Acc']],
        'Test_Recall': [mean_row['Test_Recall'], std_row['Test_Recall']],
        'Test_Precision': [mean_row['Test_Precision'], std_row['Test_Precision']],
        'Test_F1': [mean_row['Test_F1'], std_row['Test_F1']],
        'Test_AUC': [mean_row['Test_AUC'], std_row['Test_AUC']],
        'Y_true': [None, None],
        'Y_pred': [None, None]
    })

    # 合并
    df = pd.concat([df, summary], ignore_index=True)

    df.to_csv(f'{root_folder}/best_results.csv', index=False)
    pd.DataFrame([best_params_cv]).to_csv(f'{root_folder}/best_params.csv', index=False)
    if all_test_predictions:
        pd.concat(all_test_predictions, ignore_index=True).to_csv(
            f'{root_folder}/test_subject_predictions_all_folds.csv', index=False
        )
    pd.DataFrame(fold_weight_records).to_csv(
        os.path.join(root_folder, 'fold_class_weights.csv'), index=False
    )
    pd.DataFrame(fold_selection_records).to_csv(
        os.path.join(root_folder, 'fold_validation_selection.csv'), index=False
    )

    if not all_fold_attributions_func:
        return

        # 对功能模态
    print('all_fold_attributions_func:',all_fold_attributions_func)
    avg_func = normalize_attributions(all_fold_attributions_func)  # [num_nodes]
    print('avg_func:', avg_func)

    # 对结构模态
    avg_struct = normalize_attributions(all_fold_attributions_struct)  # [num_nodes]

    avg_mean = normalize_attributions(all_fold_attributions_mean)

    df_func = pd.DataFrame(avg_func)
    df_func.to_csv(f'{root_folder}/node_importance/attributions_func_avg.csv', index=False)
    df_struct = pd.DataFrame( avg_struct)
    df_struct.to_csv(f'{root_folder}/node_importance/attributions_struct_avg.csv', index=False)
    df_mean = pd.DataFrame(avg_mean)
    df_mean.to_csv(f'{root_folder}/node_importance/attributions_mean_avg.csv', index=False)

    # #writer.close()

if False and __name__ == '__main__':
    hyperparams = {
        't1_path':        'data/zhu_graymatter_tensor.npy',
        'cbf_path':      'data/zhu_cbf_aal90_new.npy',
        'bold_path':      'data/zhu_bold_tensor.npy',
        'fa_path':        'data/FA_tensor.npy',
        'fn_dir':         'data/zhu_DTI_matrices',
        'label_path':     'data/zhu_labels.npy',
        # Original diagnosis IDs: CN=0, SCD=1, MCI=2, AD=3.
        'task_labels':    (0, 1, 2, 3),
        'binary_class':   True,
        'keep_ratio':     0.5,  # 保留50%的边, 之前给的是0.5
        'threshold':      10,   # 结构连接矩阵阈值
        'batch_size':     256,
        'dropout':        None,

        # Classification-first four-class ablation: fit diagnosis labels before
        # reintroducing auxiliary objectives in a later experiment.
        'epochs_1':         [150],
        'epochs_2':          [80],
        'epochs_3':          [0],
        'lr_1':             [0.0001],
        'lr_2':             [0.00002],
        'lr_3':             [0.00001],

        'patience_1':       30,
        'patience_2':       20,
        'patience_3':       20,

        'scheduler_factor':    0.5,
        'scheduler_patience':  8,
        'scheduler_threshold': 1e-4,
        'scheduler_min_lr':    1e-6,
        'run_stage3':         False,

        'reg':            [0.0005],
        'lcont':             [0.001],
        'lothro':             [0.001],
        'cluster':            [0.05],
        'tau':              [0.07],#0.05, 0.07,
        'lambda_param':     [0.5],#0.3, 0.5, 0.7
        'lspe':             [0.01],
        'laux':             [0.01],
        'laux_cls':         [0.1],

        'hidden':         [32],
        'shared_dim':     [64],#, 64, 128， 90
        'aux_dim':        [32],
        'shared_dim':     [64],
        #'private_dim':    [32, 64, 128],
        'num_classes':    4,
        'heads':          [2],  # 可以尝试不同的头数
        'num_layers':     [2],

        'random_seed':    40,
        'patience':       50,
        'delta':          0,
        'mode':           'weighted_loss',
        # The binary-only attribution target is not meaningful for this run.
        'run_node_importance': False,
        'norm':           ['None'], # bn, ln, None
        'device':         torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'folder_name':      f'FourClass/FourClass-MultiGNN-mainloss-seed40-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    }
    task_configs = {
        'cn-scd': ((0, 1), ["CN", "SCD"]),
        'cn-mci': ((0, 2), ["CN", "MCI"]),
        'cn-ad': ((0, 3), ["CN", "AD"]),
        'scd-mci': ((1, 2), ["SCD", "MCI"]),
        'mci-ad': ((2, 3), ["MCI", "AD"]),
        'four-class': ((0, 1, 2, 3), ["CN", "SCD", "MCI", "AD"]),
    }
    parser = argparse.ArgumentParser(description='Train MultiModal GNN on a diagnosis classification task.')
    parser.add_argument('--task', choices=task_configs, default='four-class',
                        help='Diagnosis task to run (default: four-class).')
    parser.add_argument('--seed', type=int, default=40,
                        help='Random seed for both splits and model training (default: 40).')
    parser.add_argument('--run-node-importance', action='store_true',
                        help='Generate node attributions after each fold. This is disabled by default.')
    parser.add_argument('--max-grid-samples', type=int, default=8,
                        help='Maximum number of shuffled hyperparameter candidates per fold.')
    parser.add_argument('--quick', action='store_true',
                        help='Use a shorter exploratory schedule before the final full run.')
    parser.add_argument('--confirm-scale', type=float, default=None,
                        help='Run one full MCI-AD candidate with this AD class-weight scale.')
    parser.add_argument('--age-residualize', action='store_true',
                        help='Residualize imaging node features against age using training-fold CN subjects only.')
    parser.add_argument('--dimension-grid', action='store_true',
                        help='For MCI-AD, hold saved best settings fixed and grid-search only hidden/shared dimensions.')
    parser.add_argument('--stability-baseline', action='store_true',
                        help='Run one fixed MCI-AD baseline to assess convergence before further tuning.')
    parser.add_argument('--overfit-check', action='store_true',
                        help='Run one wider, low-dropout MCI-AD configuration to check train-set fit.')
    parser.add_argument('--norm', choices=('none', 'ln'), default='none',
                        help='Shared/private projection normalization: none or LayerNorm.')
    args = parser.parse_args()

    task_labels, classes = task_configs[args.task]
    hyperparams['task_labels'] = task_labels
    hyperparams['num_classes'] = len(classes)
    hyperparams['random_seed'] = args.seed
    hyperparams['run_node_importance'] = args.run_node_importance
    hyperparams['norm'] = ['None' if args.norm == 'none' else 'ln']
    hyperparams['age_residualize'] = args.age_residualize
    hyperparams['age_table_path'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_audit_master', 'master_subject_table.csv'))
    if args.age_residualize:
        hyperparams['run_stage3'] = True
    hyperparams['max_grid_samples'] = args.max_grid_samples
    if args.quick:
        hyperparams.update({
            'epochs_1': [60], 'epochs_2': [30], 'epochs_3': [20],
            'patience_1': 12, 'patience_2': 8, 'patience_3': 8,
        })
    task_tag = args.task.replace('-', '_')

    # MCI-AD preset based on the strongest previous run. Keep the current
    # validation split and weighted-loss setting unchanged; only tune model
    # and objective hyperparameters for this task.
    if args.task == 'mci-ad':
        hyperparams.update({
            'lr_1': [0.0002, 0.0003, 0.0005],
            'lr_2': [0.00005, 0.0001, 0.0002],
            'lr_3': [0.00001, 0.00002, 0.00005],
            'reg': [0.0005, 0.001],
            'lcont': [0.0, 0.01],
            'lothro': [0.0, 0.01],
            'lspe': [0.01],
            'laux': [0.01],
            'hidden': [8],
            'shared_dim': [32],
            'aux_dim': [8],
            'minority_weight_scale': [0.6, 0.75, 0.9, 1.0],
        })

    if args.confirm_scale is not None:
        if args.task != 'mci-ad':
            parser.error('--confirm-scale is only supported for --task mci-ad')
        hyperparams.update({
            'lr_1': [0.0003], 'lr_2': [0.0001], 'lr_3': [0.00002],
            'reg': [0.001], 'lcont': [0.01], 'lothro': [0.01],
            'minority_weight_scale': [args.confirm_scale],
            'max_grid_samples': 1,
        })

    if args.stability_baseline:
        if args.task != 'mci-ad':
            parser.error('--stability-baseline is only supported for --task mci-ad')
        hyperparams.update({
            'lr_1': [0.0001], 'lr_2': [0.0001], 'lr_3': [0.0001],
            'reg': [0.001], 'lcont': [0.01], 'lothro': [0.0],
            'lspe': [0.01], 'laux': [0.01], 'laux_cls': [0.1],
            'hidden': [8], 'shared_dim': [32], 'aux_dim': [8],
            'minority_weight_scale': [1.0], 'max_grid_samples': 1,
        })

    if args.overfit_check:
        if args.task != 'mci-ad':
            parser.error('--overfit-check is only supported for --task mci-ad')
        hyperparams.update({
            'lr_1': [0.0001], 'lr_2': [0.0001], 'lr_3': [0.0001],
            'reg': [0.001], 'lcont': [0.01], 'lothro': [0.0],
            'lspe': [0.01], 'laux': [0.01], 'laux_cls': [0.1],
            'hidden': [16], 'shared_dim': [64], 'aux_dim': [16],
            'minority_weight_scale': [1.0], 'max_grid_samples': 1,
            'dropout': 0.10,
        })

    if args.dimension_grid:
        if args.task != 'mci-ad':
            parser.error('--dimension-grid is only supported for --task mci-ad')
        # Saved 2026-09-01 per-fold objective settings.  The historical CSV
        # records its selected lr as 1e-4 but not stage-specific rates, so use
        # that recorded value consistently for all three stages.
        hyperparams.update({
            'lr_1': [0.0001], 'lr_2': [0.0001], 'lr_3': [0.0001],
            'reg': [0.001], 'tau': [0.07], 'lambda_param': [0.5],
            'lspe': [0.01], 'laux': [0.01], 'laux_cls': [0.1],
            'minority_weight_scale': [1.0],
            'hidden': [4, 8, 12, 16], 'shared_dim': [16, 32, 48, 64],
            'aux_dim': [8], 'tie_aux_dim_to_hidden': True,
            'fold_fixed_params': {
                1: {'lcont': 0.01, 'lothro': 0.0},
                2: {'lcont': 0.0, 'lothro': 0.0},
                3: {'lcont': 0.01, 'lothro': 0.0},
                4: {'lcont': 0.0, 'lothro': 0.01},
                5: {'lcont': 0.0, 'lothro': 0.01},
            },
            'max_grid_samples': 16,
        })

    hyperparams['folder_name'] = (
        f'results_cli/{task_tag}/MultiGNN-{task_tag}-seed{args.seed}-'
        f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    print(f'Running task: {args.task}; labels={task_labels}; seed={args.seed}')
    train_multimodal_grid_cv(hyperparams, classes)

    print(f'Completed task: {args.task}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train MultiModalGNN on complete ADNI CN-versus-MCI four-modality data.'
    )
    parser.add_argument('--seed', type=int, default=40,
                        help='Random seed for splits and training (default: 40).')
    parser.add_argument('--data-root', required=True,
                        help='Directory containing access-controlled ADNI-derived features.')
    parser.add_argument('--epochs-stage1', type=int, default=120,
                        help='Maximum main-modalities-only epochs per fold (default: 120).')
    parser.add_argument('--epochs-stage2', type=int, default=60,
                        help='Maximum four-modality fine-tuning epochs per fold (default: 60).')
    parser.add_argument('--max-grid-samples', type=int, default=1,
                        help='Number of shuffled hyperparameter candidates per fold (default: 1).')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Graphs per minibatch (default: 64).')
    parser.add_argument('--quick', action='store_true',
                        help='Run 4 stage-1 and 2 stage-2 epochs per fold as a smoke test.')
    parser.add_argument('--run-node-importance', action='store_true',
                        help='Calculate post-hoc node attributions after every fold.')
    parser.add_argument('--norm', choices=('none', 'ln'), default='none',
                        help='Shared/private projection normalization (default: none).')
    parser.add_argument('--dti-features', default='fa',
                        help='DTI features: fa (default), all, or a comma-separated subset such as FA,L1,MD.')
    parser.add_argument('--fc-mode', choices=('positive', 'signed'), default='positive',
                        help='BOLD FC: set negatives to zero (positive) or preserve signed FC features.')
    parser.add_argument('--result-dir', default=None,
                        help='Optional explicit result directory (used by reproducible ablation runs).')
    parser.add_argument('--disable-early-stopping', action='store_true',
                        help='Always run every configured epoch; use this to inspect overfitting curves.')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    adni_root = os.path.abspath(args.data_root)
    hyperparams = {
        'mat_path': os.path.join(adni_root, 'ROI_T1_fMRI_DTI_AAL90_CN_MCI_4modal_complete.mat'),
        'asl_feature_path': os.path.join(adni_root, 'ASL_CBF', 'CBF_AAL90_features.tsv'),
        'subject_table_path': os.path.join(adni_root, 'CN_MCI_4modal_complete_FN_subjects.csv'),
        'fn_dir': os.path.join(adni_root, 'FN'),
        'task_labels': (0, 1),
        'keep_ratio': 0.5,
        'dti_feature_mode': args.dti_features,
        'fc_mode': args.fc_mode,
        'batch_size': args.batch_size,
        'dropout': None,
        'epochs_1': [args.epochs_stage1],
        'epochs_2': [args.epochs_stage2],
        'epochs_3': [0],
        'lr_1': [0.0001],
        'lr_2': [0.00002],
        'lr_3': [0.00001],
        'patience_1': 25,
        'patience_2': 15,
        'patience_3': 0,
        'scheduler_factor': 0.5,
        'scheduler_patience': 8,
        'scheduler_threshold': 1e-4,
        'scheduler_min_lr': 1e-6,
        'run_stage3': False,
        'reg': [0.0005],
        'lcont': [0.001],
        'lothro': [0.001],
        'cluster': [0.05],
        'tau': [0.07],
        'lambda_param': [0.5],
        'lspe': [0.01],
        'laux': [0.01],
        'laux_cls': [0.1],
        'hidden': [32],
        'shared_dim': [64],
        'aux_dim': [32],
        'num_classes': 2,
        'heads': [2],
        'num_layers': [2],
        'random_seed': args.seed,
        'patience': 50,
        'delta': 0,
        'disable_early_stopping': args.disable_early_stopping,
        'mode': 'weighted_loss',
        'run_node_importance': args.run_node_importance,
        'norm': ['None' if args.norm == 'none' else 'ln'],
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'max_grid_samples': args.max_grid_samples,
        # Keep this short: the inherited plot function encodes all
        # hyperparameters in its PNG filename and Windows has a 260-char path
        # limit in this environment.
        'folder_name': args.result_dir or os.path.join(
            script_dir,
            'ra',
            f'adni-{args.dti_features.replace(",", "+").lower()}-{args.fc_mode}-fc-s{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        ),
    }
    if args.quick:
        hyperparams.update({
            'epochs_1': [4], 'epochs_2': [2],
            'patience_1': 4, 'patience_2': 2,
        })

    print('Running ADNI CN-versus-MCI four-modality training')
    print('Cohort: 708 complete subjects (CN=467, MCI=241)')
    train_multimodal_grid_cv(hyperparams, ['CN', 'MCI'])
    print('Completed ADNI CN-versus-MCI training')
