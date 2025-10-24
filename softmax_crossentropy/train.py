import torch


def Train(model, train_DL, criterion, optimizer,device,epoch):
    model.to(device)
    loss_history = []
    train_number = len(train_DL.dataset)

    for epoch in range(epoch):
        model.train()
        running_loss = 0
        for images, labels in train_DL:
            images = images.to(device)
            labels = labels.to(device)
            # inference
            y_hat = model(images)
            # loss
            loss = criterion(y_hat,labels)
            # update
            optimizer.zero_grad() #gradient 누적을 막기 위한 초기화
            loss.backward() # backpropagation
            optimizer.step() # update weights

            # loss accumulation
            batch_loss = loss.item() * images.shape[0] # 남는 배치 사이즈도 처리해주기 위함
            running_loss += batch_loss
        # print loss
        epoch_loss = running_loss / train_number
        loss_history += [epoch_loss]
        print(f'epoch: {epoch+1} loss: {round(epoch_loss,3)}')
        print('-'*20)

    return loss_history

