import torch
from model import ResNet
import dataset
# model load
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_model_path ='/home/user/CIFAR-10-classification/ResNet/best_models/best_model.pth'
load_model = ResNet(20).to(device)
state_dict = torch.load(best_model_path,map_location=device)
load_model.load_state_dict(state_dict)


# test
total = len(dataset.test_loader.dataset)
total_correct = 0
#total_loss = 0.0
load_model.eval()
with torch.no_grad():
    for x_batch, y_batch in dataset.test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # output logits
        y_hat = load_model(x_batch)
        # loss
        #loss = nn.CrossEntropyLoss(y_hat,y_batch)
        #batch_loss = loss.item() * x_batch.size(0)
        #total_loss += batch_loss

        # accuracy
        preds = torch.argmax(y_hat,dim=1)
        batch_correct = torch.sum(preds==y_batch).item()
        total_correct += batch_correct
test_accuracy = total_correct / total
print(f'correct: {total_correct}/{total} test_accuracy: {test_accuracy * 100:.3f}')