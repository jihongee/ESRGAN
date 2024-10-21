import numpy as np
import torch

class ExosomeDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        file_paths = [
        self.data = self.load_data(file_paths)
        self.transform = transform

    def load_data(self, file_paths):
        # 여러 파일을 읽어 데이터를 합침
        all_data = []
        for file_path in file_paths:
            data = np.loadtxt(file_path)
            all_data.append(data)

        # 리스트를 numpy 배열로 변환하여 반환
        return np.concatenate(all_data, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 데이터에 전처리 적용
        if self.transform:
            sample = self.transform(sample)

        return sample
