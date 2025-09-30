
import data_download as data
from KnnClassifier import KnnClassifier as knn
import random

indices = random.sample(range(len(data.testset)),100)
knn1 = knn(5)
knn1.data_upload(data.trainset)


correct = 0
for idx in indices:
    img, label = data.testset[idx]
    img = img.flatten()
    return_tensor = knn1.cal_distance(img)
    cand_idx = knn1.candidate_5(return_tensor)
    result = knn1.predict_one(cand_idx)

    if label == result:
        correct += 1
        print('predict image correctly')
    else:
        print('prediction is failed')

accuracy = correct / len(indices)

print(f'accuracy is {accuracy * 100:.4f}')


