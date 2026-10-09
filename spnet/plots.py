import matplotlib.pyplot as plt
import numpy as np
import os

# def plot_train_val(train_losses, val_losses, train_accs, val_accs, sub_folder, lr, reg, lcont, lothro, cluster, shared_dim, tau, lambda_param, hidden):
#     # 保存该参数组合的学习曲线
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     ax2 = ax1.twinx()  # 只创建一个额外的y轴用于准确率

#     # 在左侧Y轴绘制两个损失
#     ax1.plot(train_losses, 'b-', label='Train Loss')
#     ax1.plot(val_losses, 'r-', label='Val Loss')
#     # 在右侧Y轴绘制准确率
#     ax2.plot(train_accs, 'b--', label='Train Acc')
#     ax2.plot(val_accs, 'r--', label='Val Acc')

#     # 设置标签
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss', color='black')
#     ax2.set_ylabel('Accuracy', color='black')

#     # 合并图例
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.9, 0.9))

#     plt.title(f'lr={lr} reg={reg} lcont={lcont} lothro={lothro} cluster={cluster} shared_dim={shared_dim} tau={tau} lambda_param={lambda_param} hidden={hidden}')
#     plt.tight_layout()
#     plt.savefig(f'{sub_folder}/lr{lr}_reg{reg}_lcont{lcont}_lothro{lothro}_shared_dim{shared_dim}_tau{tau}_lambda_param{lambda_param}_hidden{hidden}.png')
#     plt.close()

def plot_train_val(train_losses, val_losses, train_accs, val_accs,
                   sub_folder, lr, reg, lcont, lothro,
                   shared_dim, tau, lambda_param, hidden, lspe, laux):
    """分别绘制 Loss 和 Accuracy 曲线"""

    # 创建保存目录
    os.makedirs(sub_folder, exist_ok=True)

    # ==== 第一张图：Loss ====
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve\nlr={lr}, reg={reg}, lcont={lcont}, lothro={lothro}, '
              f'shared_dim={shared_dim}, tau={tau}, '
              f'lambda={lambda_param}, hidden={hidden}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{sub_folder}/loss_lr{lr}_reg{reg}_lcont{lcont}_lothro{lothro}_'
                f'shared_dim{shared_dim}_tau{tau}_lambda{lambda_param}_hidden{hidden}_lspe{lspe}_laux{laux}.png')
    plt.close()

    # ==== 第二张图：Accuracy ====
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, 'b--', label='Train Balanced Acc')
    plt.plot(val_accs, 'r--', label='Val Balanced Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Balanced Accuracy Curve\nlr={lr}, reg={reg}, lcont={lcont}, lothro={lothro}, '
              f'shared_dim={shared_dim}, tau={tau}, '
              f'lambda={lambda_param}, hidden={hidden}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{sub_folder}/acc_lr{lr}_reg{reg}_lcont{lcont}_lothro{lothro}_'
                f'shared_dim{shared_dim}_tau{tau}_lambda{lambda_param}_hidden{hidden}_lspe{lspe}_laux{laux}.png')
    plt.close()

def plot_train_val_all(train_losses, val_losses, train_accs, val_accs,
                   sub_folder, lr_1, lr_2, lr_3, reg, lcont, lothro,
                   shared_dim, tau, lambda_param, hidden, lspe, laux):
    """分别绘制 Loss 和 Accuracy 曲线"""

    # 创建保存目录
    os.makedirs(sub_folder, exist_ok=True)

    # ==== 第一张图：Loss ====
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve\nlr_1={lr_1}, lr_2={lr_2}, lr_3={lr_3}, reg={reg}, lcont={lcont}, lothro={lothro}, '
              f'shared_dim={shared_dim}, tau={tau}, '
              f'lambda={lambda_param}, hidden={hidden}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{sub_folder}/loss_lr_1{lr_1}_lr_2{lr_2}_lr_3{lr_3}_reg{reg}_lcont{lcont}_lothro{lothro}_'
                f'shared_dim{shared_dim}_tau{tau}_lambda{lambda_param}_hidden{hidden}_lspe{lspe}_laux{laux}.png')
    plt.close()

    # ==== 第二张图：Accuracy ====
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, 'b--', label='Train Balanced Acc')
    plt.plot(val_accs, 'r--', label='Val Balanced Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy')
    plt.title(f'Balanced Accuracy Curve\nlr_1={lr_1}, lr_2={lr_2}, lr_3={lr_3}, reg={reg}, lcont={lcont}, lothro={lothro}, '
              f'shared_dim={shared_dim}, tau={tau}, '
              f'lambda={lambda_param}, hidden={hidden}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{sub_folder}/acc_lr_1{lr_1}_lr_2{lr_2}_lr_3{lr_3}_reg{reg}_lcont{lcont}_lothro{lothro}_'
                f'shared_dim{shared_dim}_tau{tau}_lambda{lambda_param}_hidden{hidden}_lspe{lspe}_laux{laux}.png')
    plt.close()


def plot_train_val_cont(train_cls_losses, train_cont_losses, val_losses, train_accs, val_accs,
                   sub_folder, lr, reg, lcont, lothro, cluster,
                   shared_dim, tau, lambda_param, hidden):
    """分别绘制 Loss 和 Accuracy 曲线"""

    # 创建保存目录
    os.makedirs(sub_folder, exist_ok=True)

    # ==== 第一张图：Loss ====
    plt.figure(figsize=(8, 5))
    plt.plot(train_cls_losses, 'b-', label='Train Loss')
    plt.plot(train_cont_losses, 'g-', label='Train Loss')
    # plt.plot(train_orth_losses, '--', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve\nlr={lr}, reg={reg}, lcont={lcont}, lothro={lothro}, '
              f'cluster={cluster}, shared_dim={shared_dim}, tau={tau}, '
              f'lambda={lambda_param}, hidden={hidden}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{sub_folder}/loss_lr{lr}_reg{reg}_lcont{lcont}_lothro{lothro}_'
                f'shared_dim{shared_dim}_tau{tau}_lambda{lambda_param}_hidden{hidden}.png')
    plt.close()

    # ==== 第二张图：Accuracy ====
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, 'b--', label='Train Acc')
    plt.plot(val_accs, 'r--', label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Curve\nlr={lr}, reg={reg}, lcont={lcont}, lothro={lothro}, '
              f'cluster={cluster}, shared_dim={shared_dim}, tau={tau}, '
              f'lambda={lambda_param}, hidden={hidden}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{sub_folder}/acc_lr{lr}_reg{reg}_lcont{lcont}_lothro{lothro}_'
                f'shared_dim{shared_dim}_tau{tau}_lambda{lambda_param}_hidden{hidden}.png')
    plt.close()
