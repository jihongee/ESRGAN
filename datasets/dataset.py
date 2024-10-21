import numpy as np
import torch

class ExosomeDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None):
        # file_path: 데이터 파일 경로
        self.data = self.load_data(file_path)
        self.transform = transform

    def load_data(self, file_path):
        # 텍스트 파일에서 데이터를 읽고 주석을 제거하는 전처리
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # 주석(#)으로 시작하는 줄은 무시
                if line.startswith('#'):
                    continue
                # 숫자 데이터만 읽어들여서 float 리스트로 변환
                numbers = [float(num) for num in line.split()]
                data.append(numbers)
        
        # 2D numpy array로 변환
        return np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 데이터에 전처리 적용
        if self.transform:
            sample = self.transform(sample)

        return sample
