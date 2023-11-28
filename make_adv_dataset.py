import os
import sklearn

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

import models
from models import MNIST_target_net

class LicenseNumsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path).float() / 255.0  # PNG画像を読み込む
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


use_cuda=True
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

image_nc=1
batch_size = 32

# 全学習用ファイル名をtrain_datasetからリストアップ
train_files = [os.path.join('train_dataset',f) for f in os.listdir('train_dataset')]

# 全テスト用ファイル名をtest_datasetからリストアップ
test_files = [os.path.join('test_dataset', f) for f in os.listdir('test_dataset')]

# ターゲット攻撃用事前学習済みAdvGANのモデルpathをリストアップ
models_list = [model for model in os.listdir('models/target_attack_models')]

# ファイル名からラベルを取得
train_labels = [int(f.split('_')[2]) for f in train_files]
test_labels = [int(f.split('_')[2]) for f in test_files]

train_dataset = LicenseNumsDataset(train_files, train_labels)
test_dataset = LicenseNumsDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)


for model in models_list:

    # load the generator of adversarial examples
    gen_input_nc = image_nc
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(
        os.path.join('models/target_attack_models', model)))
   
    # モデルを読み込み推論モードにする
    pretrained_G.eval()
    
    count = 0
    for i, data in enumerate(train_loader, 0):
        test_img, test_label = data
        print(test_label.shape)
        test_img, test_label = test_img.to(device), test_img.to(device)

        # 摂動を加えた敵対的サンプルを作成する
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)

        # 敵対的サンプルをadv_train_datasetディレクトリに追加
        for j in range(batch_size):
            torchvision.utils.save_image(
                adv_img[j],
                f'adv_train_dataset/advimg_{test_label[j]}_{count:08d}.png')
            count+=1

    count = 0
    for i, data in enumerate(test_loader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_img.to(device)

        # 摂動を加えた敵対的サンプルを作成する
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)

        # 敵対的サンプルをadv_test_datasetディレクトリに追加
        for j in range(batch_size):
            torchvision.utils.save_image(
                adv_img[j],
                f'advimg_{test_label[j]}_{count:08d}.png')
            count+=1