import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_pickle('/home/user/CIFAR-10-classification/ResNet/log_1/training_log.pkl')
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
plt.savefig('/home/user/CIFAR-10-classification/ResNet/log_1/train_val_loss.png')
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
plt.savefig('/home/user/CIFAR-10-classification/ResNet/log_1/train_val_accuracy.png')
plt.show()