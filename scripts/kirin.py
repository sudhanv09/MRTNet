from pickleshare import Path
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import models
from torchvision import transforms

from pathlib import Path

class MRTNet(LightningModule):
    def __init__(self, input_shape, train_path, val_path):
        super().__init__()

        self.train_dir = train_path
        self.val_dir = val_path
        self.batch_size = 16

        self.augment = transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

        self.save_hyperparameters()
        self.learning_rate = 2e-4
        self.dim = input_shape

        self.feature_extractor = models.resnet18('DEFAULT')
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


        n_sizes = self._get_conv_output(input_shape)

        self.classifier = nn.Linear(n_sizes, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='binary')

    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))


        output_feat = self._forward_features(tmp_input) 
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size
        
    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x
    
    # will be used during inference
    def forward(self, x):
       x = self._forward_features(x)
       x = x.view(x.size(0), -1)
       x = self.classifier(x)
       
       return x
    
    def training_step(self, batch):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)


        acc = self.accuracy(out, gt)


        self.log("train/loss", loss)
        self.log("train/acc", acc)


        return loss
    
    def validation_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)


        self.log("val/loss", loss)


        acc = self.accuracy(out, gt)
        self.log("val/acc", acc)


        return loss
    
    def test_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)
        
        return {"loss": loss, "outputs": out, "gt": gt}


    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        output = torch.cat([x['outputs'] for x in outputs], dim=0)
        
        gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
        self.log("test/loss", loss)
        acc = self.accuracy(output, gts)
        self.log("test/acc", acc)
        
        self.test_gts = gts
        self.test_output = output
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.ImageFolder(self.train_dir, transform=self.augment)
        self.val_dataset = torchvision.datasets.ImageFolder(self.val_dir, transform=self.transform)

        self.val_split, self.test_split = random_split(self.val_dataset, [450, 50])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size, num_workers=2)


if __name__ == '__main':

    train_path = Path('../data/train/')
    val_path = Path('../data/val/')

    model = MRTNet((3,224, 224), train_path, val_path)
    trainer = Trainer(
            callbacks=[TQDMProgressBar(refresh_rate=10)],
            accelerator = "cpu",
            max_epochs=20)

    trainer.fit(model)

    trainer.test()
