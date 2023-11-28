import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 target_num,
                 exp_num,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.target_num = target_num
        self.exp_num = exp_num
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # # C&W loss function
            # real = torch.sum(onehot_labels * probs_model, dim=1)
            # other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            # zeros = torch.zeros_like(other)
            # loss_adv = torch.max(real - other, zeros)
            # loss_adv = torch.sum(loss_adv)

            ###
            # target adversarial loss function
            ###
            target_labels = [self.target_num]*len(labels)
            target_onehot_labels = torch.eye(self.model_num_labels, device=self.device)[target_labels]

            target_conf = torch.sum(target_onehot_labels * probs_model, dim=1)

            # 最大値を取るのではなく、和を求めることでターゲット以外のラベルに対してモデルが持つ確信を失わせる
            others_conf = torch.sum((1 - target_onehot_labels) * probs_model, dim=1)
            zeros = torch.zeros_like(others_conf)

            losses_adv = torch.max(others_conf - target_conf, zeros)
            loss_adv = torch.sum(losses_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):

        self.loss_D_hist = []
        self.loss_G_fake_hist = []
        self.loss_perturb_hist = []
        self.loss_adv_hist = []
    
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            
            # for i, data in enumerate(train_dataloader, start=0):
            progress_bar = tqdm(enumerate(train_dataloader, start=0), total=len(train_dataloader))
            for i, data in progress_bar:

                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

                # tqdmの進行状況バーに情報を表示
                progress_bar.set_description(f"Epoch {epoch}/{epochs}")
                progress_bar.set_postfix(loss_D=loss_D_sum/(i+1), 
                                         loss_G_fake=loss_G_fake_sum/(i+1),
                                         loss_perturb=loss_perturb_sum/(i+1), 
                                         loss_adv=loss_adv_sum/(i+1))

            # print statistics
            num_batch = len(train_dataloader)
            # print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
            #  \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
            #       (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
            #        loss_perturb_sum/num_batch, loss_adv_sum/num_batch))
            
            self.loss_D_hist.append(loss_D_sum/num_batch)
            self.loss_G_fake_hist.append(loss_G_fake_sum/num_batch)
            self.loss_perturb_hist.append(loss_perturb_sum/num_batch)
            self.loss_adv_hist.append(loss_adv_sum/num_batch)

            # save generator
            if epoch%10 == 0:
                netG_file_name = models_path + 'netG_fake'+ str(self.target_num) + '_' + str(epoch) + 'epoch.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

            # 追加：学習曲線を描画するメソッドを呼び出し
            if epoch%10 == 0:
                self.plot_train_curves(epoch)

    def plot_train_curves(self, epoch):
        epochs = len(self.loss_D_hist)
        plt.figure(figsize=(10, 8))

        # 各損失の学習曲線を描画
        plt.plot(range(1, epochs + 1), self.loss_D_hist, label="Loss_D")
        plt.plot(range(1, epochs + 1), self.loss_G_fake_hist, label="Loss_G_fake")
        plt.plot(range(1, epochs + 1), self.loss_perturb_hist, label="Loss_perturb")
        plt.plot(range(1, epochs + 1), self.loss_adv_hist, label="Loss_adv")

        plt.title("Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.yscale('log')

        # カレントディレクトリに.png形式で保存
        plt.savefig(f"./outputs/exp{self.exp_num}_fake{self.target_num}/tr_curves/training_curves_{epoch}.png")
        plt.close()