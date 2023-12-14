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
    
    def __init__(self, config, trainset, validset, backbone, learning_rate=1e-3, weight_decay=1e-3, evalid=5) -> None:
        self.config = config
        self.trainloader = DataLoader(trainset, self.config['batch_size'], shuffle=True)
        self.validloader = DataLoader(validset, self.config['batch_size'], shuffle=False)
        self.model = CRNN(backbone, len(self.config['chars']), self.config['rnn_hidden']).apply(self.weights_init).to(device)
        self.criterion = nn.CTCLoss(blank=0)
        self.optim = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lr_scheduler = ReduceLROnPlateau(self.optim, verbose=True, patience=5)
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        self.print_eval = evalid
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
    
    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH))
    
    def save(self, PATH):
        torch.save(self.model.state_dict(), PATH)
        
    def valid(self):
        with torch.no_grad():
            self.model.zero_grad()
            total_loss = 0
            for batch in self.validloader:
                image, text = batch
                text_batch_logits = self.model(image.to(device))
                loss = self.compute_loss(text, text_batch_logits)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.validloader)
        return avg_loss
    
    def fit(self):
        prev_loss = 999
        for epoch in range(self.config['epochs']):
            print('EPOCH: %03d/%03d' % (epoch, self.config['epochs']))
            total_loss = 0
            pbar = tqdm(enumerate(self.trainloader), ncols=100)
            for idx, batch in pbar:
                loss = self.step(batch)
                total_loss += loss
                pbar.set_description("Loss: %0.5f" % (total_loss/(idx + 1)))
            avg_loss = total_loss / len(self.trainloader)
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(self.trainloader)
            
            # Eval model every print_eval epoch
            if epoch % self.print_eval == 0:
                val_loss = self.valid()
                # Save the best model
                if val_loss < prev_loss:
                    prev_loss = val_loss
                    self.save('best-model.pth')
                    print('Best model saved with loss: {}'.format(val_loss))
                
                # Create folder for save model
                folder = self.config['save_dir'] + self.now
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                print('Model saved at epoch {}'.format(epoch))
                self.save(os.path.join(folder, 'model-{}.pth'.format(epoch)))
                    
                val_acc = self.calc_accuracy(self.validloader)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                print("Val_loss: %.5f ====== Val_Acc: %.5f" % (val_loss, val_acc))
            self.lr_scheduler.step(avg_loss)
    
    def calc_accuracy(self, loader):
        correct = 0
        num_sample = 0
        with torch.no_grad():
            self.model.zero_grad()
            for batch in loader:
                image, text = batch
                text_batch_logits = self.model(image.to(device))
                preds = decode_predictions(text_batch_logits.cpu())
                for i, pred in enumerate(preds):
                    num_sample += 1
                    text_pred = correct_prediction(pred)
                    if text_pred == text[i]:
                        correct += 1
            
        return correct / num_sample
       
    def weights_init(self, m):
        classname = m.__class__.__name__
        if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
