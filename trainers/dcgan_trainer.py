import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import trange
from torch.utils.data import DataLoader, random_split
from ..models.generator import Generator
from ..models.discriminator import Discriminator
from typing import Tuple
import torchvision


class DCGAN_Trainer:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.dcgan_config = config
        self.lr = self.dcgan_config["lr"]
        self.betas = self.dcgan_config["betas"]
        self.epochs = self.dcgan_config["epochs"]
        self.min_lr = self.dcgan_config["min_lr"]
        self.lr_decay = self.dcgan_config["lr_decay"]
        self.patience = self.dcgan_config["patience"]
        self.n_layers = self.dcgan_config["n_layers"]
        self.batch_size = self.dcgan_config["batch_size"]
        self.num_features = self.dcgan_config["num_features"]
        self.latent_dim = self.dcgan_config["latent_dim"]
        self.num_channels = self.dcgan_config["num_channels"]
        
        
        self.weights_dir = "ganestro/trainers/weights"
        self.weights_gen = "ganestro/trainers/weights/generator.pth"
        self.weights_disc = "ganestro/trainers/weights/discriminator.pth"
        
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

    def train(self, X: torch.Tensor, logging_interval: int = 100) -> Tuple[Generator, Discriminator]:
        self.X = X.view(X.size(0), -1)
        self.criterion = nn.BCEWithLogitsLoss()

        self.log_dict = {'train_generator_loss_per_batch': [],
            'train_discriminator_loss_per_batch': [],
            'train_discriminator_real_acc_per_batch': [],
            'train_discriminator_fake_acc_per_batch': [],
            'images_from_noise_per_epoch': []
        }
        
        self.generator = Generator(num_features=self.num_features, 
                                   latent_dim=self.latent_dim,
                                   num_channels=self.num_channels).to(
            self.device
        )
        self.discriminator = Discriminator(num_features=self.num_features,
                                           num_channels=self.num_channels).to(
            self.device
        )
       
        # self.ae_net = AEModel(self.architecture, input_dim=self.X.shape[1]).to(
        #     self.device
        # )

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        if os.path.exists(self.weights_path):
            self.generator.load_state_dict(torch.load(self.weights_gen))
            self.discriminator.load_state_dict(torch.load(self.weights_disc))

        train_loader, valid_loader = self._get_data_loader()

        self.fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device) # format NCHW
        start_time = time.time()
        
        print("Training DCGAN:")
        for epoch in range(self.epochs):
            
            self.generator.train()
            self.discriminator.train()
            
            gen_loss = 0.0
            disc_loss = 0.0
            for batch_idx, (features, _) in enumerate(train_loader):
                
                batch_size = features.size(0)
                
                # real images
                real_images = features.to(self.device)
                real_labels = torch.ones(batch_size, device=self.device) # real label = 1

                # generated (fake) images
                noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)  # format NCHW
                fake_images = self.generator(noise)
                fake_labels = torch.zeros(batch_size, device=self.device) # fake label = 0
                flipped_fake_labels = real_labels # here, fake label = 1

                # --------------------------
                # Train Discriminator
                # --------------------------

                self.optimizer_D.zero_grad()

                # get discriminator loss on real images
                discr_pred_real = self.discriminator(real_images).view(-1) # Nx1 -> N
                real_loss = self.criterion(discr_pred_real, real_labels)
                # real_loss.backward()

                # get discriminator loss on fake images
                discr_pred_fake = self.discriminator(fake_images.detach()).view(-1)
                fake_loss = self.criterion(discr_pred_fake, fake_labels)
                # fake_loss.backward()

                # combined loss
                discr_loss = 0.5*(real_loss + fake_loss)
                discr_loss.backward()

                self.optimizer_D.step()

                # --------------------------
                # Train Generator
                # --------------------------

                self.optimizer_G.zero_grad()

                # get discriminator loss on fake images with flipped labels
                discr_pred_fake = self.discriminator(fake_images).view(-1)
                gener_loss = self.criterion(discr_pred_fake, flipped_fake_labels)
                gener_loss.backward()

                self.optimizer_G.step()

                # --------------------------
                # Logging
                # -------------------------- 

                self.log_dict['train_generator_loss_per_batch'].append(gener_loss.item())
                self.log_dict['train_discriminator_loss_per_batch'].append(discr_loss.item())
                
                predicted_labels_real = torch.where(discr_pred_real.detach() > 0., 1., 0.)
                predicted_labels_fake = torch.where(discr_pred_fake.detach() > 0., 1., 0.)
                acc_real = (predicted_labels_real == real_labels).float().mean()*100.
                acc_fake = (predicted_labels_fake == fake_labels).float().mean()*100.
                self.log_dict['train_discriminator_real_acc_per_batch'].append(acc_real.item())
                self.log_dict['train_discriminator_fake_acc_per_batch'].append(acc_fake.item())         
                
                if not batch_idx % logging_interval:
                    print('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                        % (epoch+1, self.epochs, batch_idx, 
                            len(train_loader), gener_loss.item(), discr_loss.item()))

            
            self.validate(valid_loader)
            
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
        torch.save(self.generator.state_dict(), self.weights_gen)
        torch.save(self.discriminator.state_dict(), self.weights_disc)
        
        return self.generator, self.discriminator, self.log_dict

    def validate(self) -> float:
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise).detach().cpu()
            self.log_dict['images_from_noise_per_epoch'].append(
                torchvision.utils.make_grid(fake_images, padding=2, normalize=True))
            
    def _get_data_loader(self) -> tuple:
        trainset_len = int(len(self.X) * 0.9)
        validset_len = len(self.X) - trainset_len
        trainset, validset = random_split(self.X, [trainset_len, validset_len])
        train_loader = DataLoader(
            trainset, batch_size=self.ae_config["batch_size"], shuffle=True
        )
        valid_loader = DataLoader(
            validset, batch_size=self.ae_config["batch_size"], shuffle=False
        )
        return train_loader, valid_loader