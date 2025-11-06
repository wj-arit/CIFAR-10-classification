import torch
import torch.nn as nn
import model
import dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_model = model.ResNet(20).to(device)
criterion = nn.CrossEntropyLoss()

# hyper parameters
#learning_rate = 0.01
momentum = 0.9
epoch = 160
optimizer = torch.optim.SGD(my_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

def train_one_epoch(my_model,train_dataloader,optimizer,criterion,device):
    my_model.train()
    total_correct = 0
    total_loss = 0.0
    total = len(dataset.train_dataset)
    for x_batch, y_batch in train_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # logit(output)
        y_hat = my_model(x_batch)
        # loss(cross_entropy)
        loss = criterion(y_hat, y_batch)
        # initialize gradient
        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update parameters
        optimizer.step()

        # calculate batch_loss
        batch_loss = loss.item() * x_batch.size(0)
        total_loss += batch_loss

        # prediction in train
        preds = torch.argmax(y_hat,dim=1) # indices of high probability
        batch_correct = torch.sum(preds==y_batch).item()
        total_correct += batch_correct
    epoch_loss = total_loss / total
    epoch_accuracy = total_correct / total
    #print(f'train epoch loss: {epoch_loss} train epoch accuracy: {epoch_accuracy}')

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total = len(dataloader.dataset)
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            batch_loss = loss.item() * x_batch.size(0)
            total_loss += batch_loss

            preds = torch.argmax(y_hat,1)
            correct = torch.sum(preds==y_batch).item()
            total_correct += correct

        val_loss = total_loss/total
        val_accuracy = total_correct/total
        #print(f'validation loss: {val_loss} validation accuracy: {val_accuracy}')
        return val_loss, val_accuracy
