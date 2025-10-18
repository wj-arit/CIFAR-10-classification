import torch
import data_download as data
from KnnClassifier import KnnClassifier as knn
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

indices = range(len(data.testset))
k_values = [3,5,7,15]
result_accuracies = {}
for k_value in k_values:
    knn1 = knn(k_value, device=device)
    knn1.data_upload(data.trainset)


    correct = 0
    for idx in tqdm(indices, desc=f'k={k_value}',ncols=80):
        img, label = data.testset[idx]
        return_tensor = knn1.cal_distance(img)
        cand_idx = knn1.candidate(return_tensor)
        result = knn1.predict_one(cand_idx)

        if label == result:
            correct += 1
            #print('predict image correctly')
        else:
            #print('prediction is failed')
            pass
    accuracy = correct / len(indices)
    result_accuracies[k_value] = accuracy

    print(f'k = {k_value} accuracy = {accuracy:.2f}')

best_k = max(result_accuracies,key=result_accuracies.get)
print(f'best k size = {best_k} , accuracy = {result_accuracies[best_k]:2f}')

# result graph
plt.figure(figsize=(6,4))
plt.plot(list(result_accuracies.keys()), list(result_accuracies.values()), marker='o')
plt.title("KNN Accuracy according to K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()


