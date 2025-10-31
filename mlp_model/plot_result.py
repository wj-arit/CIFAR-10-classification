import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_pickle('/home/user/CIFAR-10-classification/mlp_model/log_7(sgd,lr=0.003,w_aug,momentum=0.9,epoch=200)/training_log.pkl')
print(df.head())
print(df.info())

# plot loss graph
plt.figure(figsize=(10,6))
plt.plot(df['train_loss'], label='Train Loss', linewidth=2)
plt.plot(df['val_loss'], label='Validation Loss', linewidth=2)

plt.title('Training Loss & Validation Loss', fontsize=14)
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('/home/user/CIFAR-10-classification/mlp_model/log_7(sgd,lr=0.003,w_aug,momentum=0.9,epoch=200)/loss_plot(sgd,lr=0.001,momentum=0.9,w_aug,epoch=200).png')
plt.show()

#plot accuracy graph
plt.figure(figsize=(10,6))
plt.plot(df['train_accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(df['val_accuracy'], label='Validation Accuracy', linewidth=2)

plt.title('Train Accuracy & Validation Accuracy', fontsize=14)
plt.xlabel('Epoch',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('/home/user/CIFAR-10-classification/mlp_model/log_7(sgd,lr=0.003,w_aug,momentum=0.9,epoch=200)/accuracy_plot(sgd,lr=0.001,momentum=0.9,w_aug,epoch=200).png')
plt.show()