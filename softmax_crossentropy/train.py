import numpy as np
import torch
from model import MLP_CIFAR10
from dataset import train_DS
import torch.nn as nn
from k_fold_validation import k_fold_train
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {DEVICE} device")

model_class = MLP_CIFAR10
full_dataset = train_DS
criterion = nn.CrossEntropyLoss()
device = DEVICE
optimizer_class = lambda params: torch.optim.SGD(params,lr=0.01,momentum=0.8)
epoch = 200
k_size = 5
batch_size = 64
patience = 20
fold_results = k_fold_train(
                            model_class,train_DS,criterion,device,
                            optimizer_class,epoch,k_size,batch_size,patience
                            )

# fold_results: k_fold_train()의 반환값 (각 fold의 train_losses, val_losses 포함)
# 예시: fold_results = [{'train_losses': [...], 'val_losses': [...]}, {...}, ...]

# 가장 긴 epoch 길이를 기준으로 배열 초기화
for i, r in enumerate(fold_results):
    print(f"Fold {i}: keys = {list(r.keys())}")

max_epoch = max(len(r["train_losses"]) for r in fold_results)
avg_train = np.zeros(max_epoch)
avg_val = np.zeros(max_epoch)
count_train = np.zeros(max_epoch)  # 각 epoch별로 몇 개 fold가 기여했는지
count_val = np.zeros(max_epoch)

for r in fold_results:
    length = len(r["train_losses"])
    avg_train[:length] += r["train_losses"]
    avg_val[:length] += r["val_losses"]
    count_train[:length] += 1
    count_val[:length] += 1

# 0으로 나누는 걸 방지하면서 fold별로 실제 참여한 수만 나누기
avg_train = np.divide(avg_train, count_train, out=np.zeros_like(avg_train), where=count_train != 0)
avg_val = np.divide(avg_val, count_val, out=np.zeros_like(avg_val), where=count_val != 0)

plt.figure(figsize=(8,5))
plt.plot(avg_train, label='Train Loss (avg)', color='tab:blue', linewidth=2)
plt.plot(avg_val, label='Validation Loss (avg)', color='tab:orange', linewidth=2)
plt.title("Average K-Fold Training vs Validation Loss", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('loss_curve.png',dpi=300)
plt.show()
