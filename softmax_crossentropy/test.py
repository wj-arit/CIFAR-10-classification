import torch
from model import MLP_CIFAR10
from dataset import test_DL

def Test(model, test_DL):
    model.eval()
    with torch.no_grad():
        rcorrect = 0
        for x_batch, y_batch in test_DL:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            # inference
            y_hat = model(x_batch)
            # accuracy accumulation
            pred = y_hat.argmax(dim=1)
            corrects_b = torch.sum(pred == y_batch).item()
            rcorrect += corrects_b
        accuracy_e = rcorrect/len(test_DL.dataset)*100
    print(f"Test accuracy: {rcorrect}/{len(test_DL.dataset)} ({round(accuracy_e,1)} %)")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
save_model_path_1 = './optimizer_SGD/4layer(m+0.01+w_aug)/best_model_fold1.pt'
save_model_path_2 = './optimizer_SGD/4layer(m+0.01+w_aug)/best_model_fold2.pt'
save_model_path_3 = './optimizer_SGD/4layer(m+0.01+w_aug)/best_model_fold3.pt'
save_model_path_4 = './optimizer_SGD/4layer(m+0.01+w_aug)/best_model_fold4.pt'
save_model_path_5 = './optimizer_SGD/4layer(m+0.01+w_aug)/best_model_fold5.pt'
load_model = MLP_CIFAR10().to(DEVICE)

# eval fold1
load_model.load_state_dict(torch.load(save_model_path_1, map_location=DEVICE))
Test(load_model, test_DL)

# eval fold2
load_model.load_state_dict(torch.load(save_model_path_2, map_location=DEVICE))
Test(load_model, test_DL)

# eval fold3
load_model.load_state_dict(torch.load(save_model_path_3, map_location=DEVICE))
Test(load_model, test_DL)

# eval fold4
load_model.load_state_dict(torch.load(save_model_path_4, map_location=DEVICE))
Test(load_model, test_DL)

# eval fold5
load_model.load_state_dict(torch.load(save_model_path_5, map_location=DEVICE))
Test(load_model, test_DL)

# map_location 있어야 GPU로 학습했던 거 현재 device로 불러올 수 있음
# map_location 있어야 GPU로 학습했던 거 현재 device로 불러올 수 있음

def count_params(model):
    num = sum([p.numel() for p in model.parameters() if model.requires_grad_])
    return num

print(count_params(load_model))