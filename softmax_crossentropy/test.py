import torch

from knn_classification.result import accuracies
from model import MLP_CIFAR10
from datasets import test_DL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP_CIFAR10().to(DEVICE)

model.load_state_dict(torch.load('mlp_cifar10.pt',map_location=DEVICE))
model.eval()

correct,total = 0, 0
with torch.no_grad():
    for images, labels in test_DL:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        y_hat = model(images)
        pred = y_hat.argmax(dim=1)
        correct = torch.sum(pred == labels).item()
        total += correct
    accuracies = total/len(test_DL.dataset()) * 100
    
print(f'Test accuracy: {total}/{len(test_DL.dataset())}, {accuracies}%')