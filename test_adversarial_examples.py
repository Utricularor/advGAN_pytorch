import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
from sklearn.metrics import confusion_matrix
import numpy as np


use_cuda=True
image_nc=1
batch_size = 256

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "./MNIST_target_model_256128size.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_256128_epoch_250.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
all_true_labels = []
all_pred_labels = []

# test adversarial examples in MNIST training dataset
# 画像の変換を定義
transform = transforms.Compose([
    transforms.Resize((256, 128)),  # 画像のサイズを256x128に変更
    transforms.ToTensor()           # 画像をテンソルに変換
])
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transform, download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
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

print('MNIST training dataset:')
print('num_all: ', len(mnist_dataset))
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))

conf_matrix_train = confusion_matrix(all_true_labels, all_pred_labels)
print("Training set confusion matrix:")
print(conf_matrix_train)


all_true_labels = []
all_pred_labels = []

# test adversarial examples in MNIST testing dataset
mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transform, download=True)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
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

    if i % 1000 == 0:
        for j in range(10):
            sample = adv_img[j]
            sample_label = test_label[j]
            torchvision.utils.save_image(sample, f'outputs/adv_imgs/20231003_256128_250ep_ans{sample_label}_pred{pred_lab[j]}.png')

print('num_all: ', len(mnist_dataset_test))
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))
conf_matrix_test = confusion_matrix(all_true_labels, all_pred_labels)
print("Testing set confusion matrix:")
print(conf_matrix_test)


