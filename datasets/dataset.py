import numpy as np
import torch
from skimage.transform import rotate  # 이미지 회전 함수

class ExosomeDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None, augment=False):
        self.file_paths = file_paths
        self.data = self.load_data(file_paths)
        self.transform = transform
        self.augment = augment  # 증강 여부 설정

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

    def augment_image(self, image):
        augmented_images = []

        # Rotation and Flipping
        for angle in [0, 90, 180, 270]:
            rotated = rotate(image, angle, reshape=False)

            # 중복된 이미지가 아닌 경우 추가
            if not any(np.array_equal(rotated, img) for img in augmented_images):
                augmented_images.append(rotated)

            # 좌우 반전
            flipped_lr = np.fliplr(rotated)
            if not any(np.array_equal(flipped_lr, img) for img in augmented_images):
                augmented_images.append(flipped_lr)

            # 상하 반전
            flipped_ud = np.flipud(rotated)
            if not any(np.array_equal(flipped_ud, img) for img in augmented_images):
                augmented_images.append(flipped_ud)

            # 최대 8개의 고유한 이미지로 제한
            if len(augmented_images) >= 8:
                break

        return augmented_images[:8]  # 8개의 이미지만 반환

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 데이터 증강 적용
        if self.augment:
            sample = self.augment_image(sample)

        # 데이터에 추가적인 전처리 적용
        if self.transform:
            sample = self.transform(sample)

        return sample
