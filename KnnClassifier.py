import torch
import data_download
from data_download import img_train


class KnnClassifier:
    def __init__(self, k_num : int) -> None:
        self.k_num = k_num
        self.img_train = None
        self.label_train = None

    def data_upload(self, dataset: torch.Tensor) -> None:
        self.img_train = torch.stack([img.flatten() for img, _ in dataset])
        self.label_train = torch.tensor([label for _, label in dataset])

    def cal_distance(self, input_img: torch.Tensor) -> torch.Tensor:
        input_img = input_img.flatten()
        result_tensor = torch.norm(self.img_train - input_img, p = 2, dim = 1)
        return result_tensor

    def candidate(self, result: torch.Tensor) -> torch.Tensor:
        cand_idx = torch.topk(result,k = self.k_num,largest = False).indices
        return cand_idx

    def predict_one(self, idx: torch.Tensor) -> int:
        predict_labels = self.label_train[idx]
        values, counts = torch.unique(predict_labels, return_counts = True)
        prediction = values[torch.argmax(counts)]
        return prediction

