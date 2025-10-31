import torch
import torch.nn as nn
import model
import dataset
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
my_model = model.MlpModel().to(device)
criterion = nn.CrossEntropyLoss()

# hyper parameters
learning_rate = 0.003
momentum = 0.9
epoch = 200
optimizer = torch.optim.SGD(my_model.parameters(),lr=learning_rate,momentum=momentum)

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




if __name__ == '__main__':
    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    best_val_loss = [0, float('inf')]
    count = 0
    for epo in range(4):
        if epo == 0:
            print(f'device: {device}')
        # best val_loss compare
        train_loss, train_accuracy = train_one_epoch(
            my_model,
            train_dataloader=dataset.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        val_loss, val_accuracy = validate_one_epoch(
            my_model,
            dataloader=dataset.val_loader,
            criterion=criterion,
            device=device
        )
        # result saving
        history['epoch'].append(epo + 1)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        # extract best model
        if val_loss <= best_val_loss[1]:
            best_val_loss[0] = epo + 1
            best_val_loss[1] = val_loss
            count += 1

            # save model
            save_model_path = f'/home/user/CIFAR-10-classification/mlp_model/best_models/best_model_epoch_{epo + 1}.pht'
            torch.save(my_model.state_dict(), save_model_path)
            print('save model complete')
            print(f'model change number: {count}')
            print(val_loss)
    df = pd.DataFrame(history)
    df.to_pickle('training_log.pkl')
    df.to_csv('training_log.csv', index=False)
    print('history saving complete')





