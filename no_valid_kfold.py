import os
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
from tqdm import tqdm

# 全局 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True

# === Dataset 全部载入版本 ===
class EEGDataset(Dataset):
    def __init__(self, x_all, y_all, indices):
        self.x_all   = x_all
        self.y_all   = y_all
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (
            torch.from_numpy(self.x_all[i].astype(np.float32)),
            torch.tensor(self.y_all[i], dtype=torch.float32),
        )

# === 简单线性模型 ===
class LRModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.linear(x)

# === 指标计算 ===
def compute_metrics(cm, label):
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if tp + fn > 0 else 0
    spec = tn / (tn + fp) if tn + fp > 0 else 0
    ppv  = tp / (tp + fp) if tp + fp > 0 else 1
    npv  = tn / (tn + fn) if tn + fn > 0 else 1
    acc  = (tp + tn) / (tp + tn + fp + fn)
    tp, tn, fp, fn = map(np.float64, (tp, tn, fp, fn))
    denom = np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom != 0 else 0
    print(f"\n=== {label} ===")
    print(f"Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, "
          f"PPV: {ppv:.4f}, NPV: {npv:.4f}, MCC: {mcc:.4f}, Accuracy: {acc:.4f}")
    print(cm)
    return sens, spec, ppv, npv, mcc, acc

def show_class_distribution(name, indices, y):
    unique, counts = np.unique(y[indices], return_counts=True)
    print(f"{name} class distribution: {dict(zip(unique, counts))}")

# === 训练（无 validation），固定阈值 0.5 ===
def train_fold_no_valid(x_all, y_all, subject_ids,
                        train_idx, test_idx,
                        batch_size=64, lr=1e-3, wd=1e-3, max_epochs=25):

    # 模型 & 优化器
    input_dim = x_all.shape[1]
    model = LRModel(input_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    # DataLoader
    train_loader      = DataLoader(EEGDataset(x_all, y_all, train_idx),
                                   batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(EEGDataset(x_all, y_all, train_idx),
                                   batch_size=batch_size, shuffle=False)
    test_loader       = DataLoader(EEGDataset(x_all, y_all, test_idx),
                                   batch_size=batch_size, shuffle=False)

    # 损失（考虑类别不平衡）
    pos_weight = torch.tensor(
        [(y_all[train_idx]==0).sum() / (y_all[train_idx]==1).sum()],
        dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 训练 Loop
    train_losses = []
    model.train()
    for epoch in range(max_epochs):
        total_loss, iters = 0.0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", dynamic_ncols=True)
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.unsqueeze(1).to(device)
            opt.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            iters += 1
            loop.set_postfix(loss=loss.item(), avg_loss=total_loss/iters)

        avg_loss = total_loss / iters
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} ─ lr = {opt.param_groups[0]['lr']:.6g}")

    # 固定阈值
    best_t = 0.5

    # Sample-Level 评估
    def evaluate_with_fixed_t(loader, indices):
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten())
        probs = np.concatenate(probs)
        preds = (probs >= best_t).astype(int)
        return confusion_matrix(y_all[indices], preds)

    # Subject-Level 评估
    def evaluate_subject(loader, indices):
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten())
        probs = np.concatenate(probs)
        sids = subject_ids[indices]
        y_sub_true, y_sub_pred = [], []
        for sid in np.unique(sids):
            mask = (sids == sid)
            y_sub_true.append(int(round(y_all[indices][mask].mean())))
            y_sub_pred.append(int((probs[mask].mean() >= best_t)))
        return confusion_matrix(y_sub_true, y_sub_pred)

    cm_tr     = evaluate_with_fixed_t(train_eval_loader, train_idx)
    cm_te     = evaluate_with_fixed_t(test_loader,       test_idx)
    cm_sub_tr = evaluate_subject(train_eval_loader, train_idx)
    cm_sub_te = evaluate_subject(test_loader,       test_idx)

    return cm_tr, cm_te, cm_sub_tr, cm_sub_te, train_losses

# === 主流程 ===
if __name__ == "__main__":
    # 加载数据
    x_train = np.load(r"D:\chen_mr_feature\SKNA_features_beat_sym_0.5_150hz_l1.44s\train_scaled.npy")
    x_test  = np.load(r"D:\chen_mr_feature\SKNA_features_beat_sym_0.5_150hz_l1.44s\test_scaled.npy")
    x_all   = np.concatenate([x_train, x_test], axis=0)

    def load_labels_from_tsv(folder_path):
        paths = glob.glob(os.path.join(folder_path, '*.tsv'))
        def sort_key(p):
            m = re.match(r'(\d+)', os.path.basename(p))
            return int(m.group(1)) if m else float('inf')
        paths = sorted(paths, key=sort_key)
        labels, counts = [], []
        for p in paths:
            v = np.loadtxt(p, delimiter='\t', usecols=0)
            labels.append(v); counts.append(len(v))
        return np.concatenate(labels), np.array(counts)

    y_tr, tr_counts = load_labels_from_tsv(r"F:\chen\ECG\SKNA_features_beat_sym_0.5_150hz_l1.44s\train")
    y_te, te_counts = load_labels_from_tsv(r"F:\chen\ECG\SKNA_features_beat_sym_0.5_150hz_l1.44s\test")
    y_all = np.concatenate([y_tr, y_te])

    # 构造 subject_ids
    n_tr = len(tr_counts); n_te = len(te_counts)
    subject_ids = np.concatenate([
        np.repeat(np.arange(n_tr), tr_counts),
        np.repeat(np.arange(n_tr, n_tr+n_te), te_counts)
    ])

    final_cms = {'s_tr':np.zeros((2,2),int),'s_te':np.zeros((2,2),int),
                 'sub_tr':np.zeros((2,2),int),'sub_te':np.zeros((2,2),int)}
    all_train_losses = []

    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(
            gkf.split(x_all, y_all, groups=subject_ids), 1):
        print(f"\n=== Fold {fold} ===")
        show_class_distribution("Train", tr_idx, y_all)
        show_class_distribution("Test",  te_idx, y_all)

        cm_tr, cm_te, cm_sub_tr, cm_sub_te, tr_loss = train_fold_no_valid(
            x_all, y_all, subject_ids, tr_idx, te_idx
        )

        compute_metrics(cm_tr,    "Train Sample-Level")
        compute_metrics(cm_te,    "Test  Sample-Level")
        compute_metrics(cm_sub_tr,"Train Subject-Level")
        compute_metrics(cm_sub_te,"Test  Subject-Level")

        final_cms['s_tr']   += cm_tr
        final_cms['s_te']   += cm_te
        final_cms['sub_tr'] += cm_sub_tr
        final_cms['sub_te'] += cm_sub_te
        all_train_losses.append(tr_loss)

    # 平均训练损失曲线
    min_len = min(len(l) for l in all_train_losses)
    avg_loss = [sum(f[i] for f in all_train_losses)/5 for i in range(min_len)]
    plt.plot(avg_loss); plt.title("Avg Train Loss (5 folds)")
    plt.xlabel("Epoch"); plt.grid(True); plt.tight_layout(); plt.show()

    # 汇总结果
    print("\n=== 🎯 Final Results ===")
    compute_metrics(final_cms['s_tr'],   "Total Train Sample-Level")
    compute_metrics(final_cms['s_te'],   "Total Test Sample-Level")
    compute_metrics(final_cms['sub_tr'], "Total Train Subject-Level")
    compute_metrics(final_cms['sub_te'], "Total Test Subject-Level")
