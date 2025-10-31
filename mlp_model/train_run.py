import train
import dataset
import pandas as pd
import torch


history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
best_val_loss = [0,float('inf')]
count = 0

#training
for epo in range(train.epoch):
    if epo == 0:
        print(f'device: {train.device}')
    # best val_loss compare
    train_loss, train_accuracy = train.train_one_epoch(
        my_model=train.my_model,
        train_dataloader=dataset.train_loader,
        optimizer=train.optimizer,
        criterion=train.criterion,
        device=train.device
    )

    val_loss, val_accuracy = train.validate_one_epoch(
        model=train.my_model,
        dataloader=dataset.val_loader,
        criterion=train.criterion,
        device=train.device
    )
    # result saving
    history['epoch'].append(epo+1)
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)

    print('-'*80)
    print(f'epoch: {epo+1} train_loss: {train_loss:4f} val_loss: {val_loss:4f} train_accuracy: {train_accuracy:4f} val_accuracy: {val_accuracy:4f}')

    # extract best model
    if val_loss <= best_val_loss[1]:
        best_val_loss[0] = epo+1
        best_val_loss[1] = val_loss
        count += 1

        #save model
        save_model_path = f'/home/user/CIFAR-10-classification/mlp_model/best_models/best_model_epoch_{epo+1}.pth'
        torch.save(train.my_model.state_dict(),save_model_path)
        print(f'model saved (epoch={epo+1}, count={count}, val_loss={val_loss:4f})')
print(f'best model: epoch={best_val_loss[0]} best_loss={best_val_loss[1]}')
df = pd.DataFrame(history)
df.to_pickle('training_log.pkl')
df.to_csv('training_log.csv',index=False)
print('history saving complete')