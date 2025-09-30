
import data_download as data
from KnnClassifier import KnnClassifier as knn
import random

indices = random.sample(range(len(data.testset)),100)
k_values = [3,5,7,15]
result_accuracies = {}
for k_value in k_values:
    knn1 = knn(k_value)
    knn1.data_upload(data.trainset)


    correct = 0
    for idx in indices:
        img, label = data.testset[idx]
        img = img.flatten()
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


