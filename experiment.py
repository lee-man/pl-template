'''
The file for generic experiment settings for classfication tasks
'''
import torch
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy



class ClsExperiment(pl.LightningModule):
    '''
    Generic PyTorch-Lightning model for classfication task.
    '''
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.params = kwargs
        
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.params['lr'],
                            momentum=self.params['momentum'], weight_decay=eval(self.params['weight_decay']))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=self.params['scheduler_gamma'])
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        _, predicted = outputs.max(1)
        acc = accuracy(predicted, targets)
        metrics = {'loss': loss, 'acc': acc}
        self.log_dict(metrics)
        # self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metrics
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        val_loss = F.cross_entropy(outputs, targets)
        _, predicted = outputs.max(1)
        acc = accuracy(predicted, targets)
        # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics)

    
    
    