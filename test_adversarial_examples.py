import os
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torchvision.datasets
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
from sklearn.metrics import confusion_matrix
import numpy as np

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

# 引数をパースする
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=250, help='the number of epoch')
args = parser.parse_args()

epoch = args.epoch

use_cuda=True
image_nc=1
batch_size = 32

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "models/target_model/MNIST_target_model_256128size.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# load the generator of adversarial examples
# pretrained_generator_path = f'./models/netG_256128_epoch_{epoch}.pth'
pretrained_generator_path = f'./outputs/exp12_fake9/models/netG_256128_fake9_epoch500.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
all_true_labels = []
all_pred_labels = []

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


num_correct = 0
for i, data in enumerate(train_loader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    all_true_labels.extend(test_label.cpu().numpy())
    all_pred_labels.extend(pred_lab.cpu().numpy())
    num_correct += torch.sum(pred_lab==test_label,0)

print('licenseNums training dataset:')
print('num_all: ', len(train_loader.dataset))
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(train_loader.dataset)))

conf_matrix_train = confusion_matrix(all_true_labels, all_pred_labels)
print("Training set confusion matrix:")
print(conf_matrix_train)

# 期待する出力のヘッダーを印刷
header = "入力データ番号 | " + " | ".join(map(str, range(10)))
print(header)

all_true_labels = []
all_pred_labels = []

num_correct = 0

for i, data in enumerate(val_loader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)

    # 摂動生成
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)

    # 摂動付加
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)

    # 敵対的サンプルに対する予測
    outputs = target_model(adv_img)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    pred_lab = torch.argmax(probabilities, 1)

    all_true_labels.extend(test_label.cpu().numpy())
    all_pred_labels.extend(pred_lab.cpu().numpy())
    num_correct += torch.sum(pred_lab==test_label,0)

    # 各入力に対する確信度を出力
    for j in range(test_img.size(0)):
        confidences = ["{:.2f}".format(prob) for prob in probabilities[j].detach().cpu().numpy()]
        print("                  {}             | (right) {} | (adv) {} | {}".format(j, test_label[j], pred_lab[j], " | ".join(confidences)))
    
    # for j in range(batch_size):
        # ターゲットラベルを９に設定しているときに５に，より間違えやすい（実験12）
        # その原因を探るべくラベル１の画像を９に間違えるように生成した摂動がどのようになった結果５と誤識別するようになったのか確認する
        # if (test_label[j] == 1) & (pred_lab[j] == 5):
        #     sample = adv_img[j]
        #     sample_label = test_label[j]
        #     torchvision.utils.save_image(test_img[j], f'./outputs/exp12_fake9/adv_imgs/ans{sample_label}_pred{pred_lab[j]}_real.png')
        #     torchvision.utils.save_image(sample, f'./outputs/exp12_fake9/adv_imgs/ans{sample_label}_pred{pred_lab[j]}_fake.png')
    
    # if i % 1000 == 0:
         # for j in range(10):
             # sample = adv_img[j]
             # sample_label = test_label[j]
             # torchvision.utils.save_image(test_img[j], f'./outputs/exp12_fake9/adv_imgs/ans{sample_label}_pred{pred_lab[j]}_real.png')
             # torchvision.utils.save_image(sample, f'./outputs/exp12_fake9/adv_imgs/ans{sample_label}_pred{pred_lab[j]}_fake.png')

print('licenseNums validation dataset:')
print('num_all: ', len(val_loader.dataset))
print('num_correct: ', num_correct.item())
conf_matrix_test = confusion_matrix(all_true_labels, all_pred_labels)
print("Validation set confusion matrix:")
print(conf_matrix_test)
print('accuracy of adv imgs in validation set: \n%f\n'%(num_correct.item()/len(val_loader.dataset)))


