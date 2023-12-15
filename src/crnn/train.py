import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import CRNN
from datasets import IDDataset
from utils import *
from torchvision.models.resnet import resnet18


class Train:
    
    def __init__(self, config) -> None:
        self.config = config
        self.model = None
        self.criterion = None
        self.optim = None
        
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        self.now = "{:%m-%d-%Y-%H-%M-%S}".format(datetime.now())
    
    def step(self, batch):
        image, text = batch
        self.optim.zero_grad()
        text_batch_logits = self.model(image.to(device))
        loss = self.compute_loss(text, text_batch_logits)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_norm'])
        self.optim.step()
        return loss.item()
        
    def valid(self, validloader):
        num_samples = len(validloader)
        with torch.no_grad():
            self.model.zero_grad()
            total_loss = 0
            for batch in validloader:
                image, text = batch
                text_batch_logits = self.model(image.to(device))
                loss = self.compute_loss(text, text_batch_logits)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_samples
        return avg_loss
    
    def compute_loss(self, text_batch, text_batch_logits):
        """
        text_batch: list of strings of length equal to batch size
        text_batch_logits: Tensor of size([T, batch_size, num_classes])
        """
        text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]
        text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                        fill_value=text_batch_logps.size(0),
                                        dtype=torch.int32).to(device) # [batch_size]
        text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
        loss = self.criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

        return loss
    
    def fit(self, model, trainloader, validloader, criterion, optimizer):
        # init
        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, verbose=True, patience=5)
        prev_loss = 999
        
        # Training process
        for epoch in range(self.config['epochs']):
            print('EPOCH: %03d/%03d' % (epoch, self.config['epochs']))
            total_loss = 0
            pbar = tqdm(enumerate(trainloader), ncols=100)
            for idx, batch in pbar:
                loss = self.step(batch)
                total_loss += loss
                pbar.set_description("Loss: %0.5f" % (total_loss/(idx + 1)))
            avg_loss = total_loss / len(trainloader)

            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(trainloader)
            
            # Eval model every print_eval epoch
            val_loss = self.valid(validloader)
            # Create folder for save model
            pathdir = os.path.join(self.config['save_dir'], self.now)
            if not os.path.isdir(self.config['save_dir']):
                os.mkdir(self.config['save_dir'])
                os.mkdir(pathdir)
            if not os.path.isdir(pathdir):
                os.mkdir(pathdir)
            print('Model saved at epoch {}'.format(epoch))
            self.save(os.path.join(pathdir, 'model-{}.pth'.format(epoch)))
            # Save the best model
            if val_loss < prev_loss:
                prev_loss = val_loss
                self.save(os.path.join(pathdir, 'best-model.pth'))
                print('Best model saved with loss: {}'.format(val_loss))
                
            val_acc = self.calc_accuracy(validloader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            print("Val_loss: %.5f ====== Val_Acc: %.5f" % (val_loss, val_acc))
            lr_scheduler.step(avg_loss)
    
    def calc_accuracy(self, loader):
        correct = 0
        num_sample = 0
        with torch.no_grad():
            self.model.zero_grad()
            for batch in loader:
                image, text = batch
                text_batch_logits = self.model(image.to(device))
                preds = decode_predictions(text_batch_logits.cpu())
                for i, _batch in enumerate(zip(preds, text)):
                    num_sample += 1
                    pred, t = _batch
                    text_pred = correct_prediction(pred)
                    if text_pred == t:
                        correct += 1
            
        return correct / num_sample

    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH))
    
    def save(self, PATH):
        torch.save(self.model.state_dict(), PATH)