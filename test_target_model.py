import os
# from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torchvision.io import read_image
# import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import torch.nn.functional as F
from models import  MNIST_target_net
# import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
# import numpy as np

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

if __name__ == "__main__":
    use_cuda = True
    image_nc = 1
    batch_size = 128

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # 画像の変換を定義
    transform = transforms.Compose([
        transforms.Resize((256, 128), antialias=True)  # 画像のサイズを256x128に変更
        # transforms.ToTensor()           # 画像をテンソルに変換
    ])

    # licenseNums_archive内の全てのファイルをリストアップ
    all_files = [f for f in os.listdir('licenseNums_archive') if os.path.isfile(os.path.join('licenseNums_archive', f))]

    # ファイル名からラベルを取得
    all_labels = [int(f.split('_')[1]) for f in all_files]

    train_files, val_files, train_labels, val_labels = train_test_split(all_files, all_labels, test_size=0.3)
    
    val_dataset = LicenseNumsDataset(val_files, val_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    pretrained_model = "./MNIST_target_model_256128size.pth"
    target_model = MNIST_target_net().to(device)
    target_model.load_state_dict(torch.load(pretrained_model))
    target_model.eval()

    # MNIST test dataset
    # mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transform, download=True)
    # licenseNums_dataset = LicenseNumsDataset(root_dir='./licenseNums_archive', transform=transforms.ToTensor())
    # test_dataloader = DataLoader(licenseNums_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    # 期待する出力のヘッダーを印刷
    header = "入力データ番号 | " + " | ".join(map(str, range(10)))
    print(header)


    all_targets = []
    all_preds = []
    
    num_correct = 0
    for i, data in enumerate(val_loader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        outputs = target_model(test_img)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        pred_lab = torch.argmax(probabilities, 1)

        num_correct += torch.sum(pred_lab==test_label,0)

        all_targets.extend(test_label.cpu().numpy())
        all_preds.extend(pred_lab.cpu().numpy())

        # 各入力に対する確信度を印刷
        for j in range(test_img.size(0)):
            confidences = ["{:.2f}".format(prob) for prob in probabilities[j].detach().cpu().numpy()]
            print("                  {}           | {} | {} | {}".format(j,test_label[j], pred_lab[j]," | ".join(confidences)))

    print('accuracy in validation set: %f\n'%(num_correct.item()/len(val_loader.dataset)))

    # Compute the confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print('Confusion Matrix:\n', cm)
