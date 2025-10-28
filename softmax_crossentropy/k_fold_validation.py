import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import numpy as np

def k_fold_train(model_class, full_dataset, criterion, device, optimizer_class,
                 epoch, k_size=5, batch_size=64, patience=20):

    k_fold = KFold(n_splits=k_size, shuffle=True, random_state=1)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(full_dataset)):
        print('-' * 30 + f' FOLD {fold+1} ' + '-' * 30),

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_DL = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_DL = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_class().to(device)
        optimizer = optimizer_class(model.parameters())

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        train_losses = []
        val_losses = []
        val_accuracies = []

        for epo in range(epoch):
            model.train()
            epoch_loss = 0.0
            for images, labels in train_DL:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)

            epoch_loss /= len(train_DL.dataset)
            train_losses.append(epoch_loss)

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for images, labels in val_DL:
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)
                    loss = criterion(output, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = output.argmax(dim=1)
                    correct += (preds == labels).sum().item()

            val_loss /= len(val_DL.dataset)
            acc = correct / len(val_DL.dataset)
            val_losses.append(val_loss)
            val_accuracies.append(acc)

            print(f"Fold {fold+1} | Epoch {epo+1}/{epoch} | "
                  f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f}")

            # --- Early Stopping check ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()  # 가중치 저장
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epo+1} (no improvement for {patience} epochs)")
                break

        # fold 종료 후 best 모델 성능 기록
        fold_results.append({
           "fold": fold + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_val_loss": best_val_loss,
            "accuracy": val_accuracies[-1] if val_accuracies else 0.0
        })

        #  모델 저장
        torch.save(best_model_state, f"best_model_fold{fold+1}.pt")

    # 전체 평균 요약
    avg_loss = np.mean([r["best_val_loss"] for r in fold_results])
    avg_acc = np.mean([r["accuracy"] for r in fold_results])
    print("\n=== K-FOLD SUMMARY ===")
    print(f"Average Validation Loss: {avg_loss:.4f}")
    print(f"Average Accuracy:        {avg_acc:.4f}")

    return fold_results

