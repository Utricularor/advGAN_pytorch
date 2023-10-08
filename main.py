import os
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torchvision.io import read_image
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
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
        image = read_image('licenseNums_archive/'+image_path).float() / 255.0  # PNG画像を読み込む
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

use_cuda=True
image_nc=1
epochs = 500
batch_size = 312
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./MNIST_target_model_256128size.pth"
targeted_model = MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# 画像の変換を定義
transform = transforms.Compose([
    transforms.Resize((256, 128), antialias=True)  # 画像のサイズを256x128に変更
    # transforms.ToTensor()           # 画像をテンソルに変換
])

# licenseNums_archive内の全てのファイルをリストアップ
all_files = [f for f in os.listdir('licenseNums_archive') if os.path.isfile(os.path.join('licenseNums_archive', f))]

# ファイル名からラベルを取得
all_labels = [int(f.split('_')[1]) for f in all_files]

# train_test_splitを使用してデータセットを分割
train_files, val_files, train_labels, val_labels = train_test_split(all_files, all_labels, test_size=0.3)

# カスタムのDatasetクラスを使用してDataLoaderを作成
train_dataset = LicenseNumsDataset(train_files, train_labels, transform=transform)
val_dataset = LicenseNumsDataset(val_files, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

advGAN.train(train_loader, epochs)
