import pytorch_lightning as pl
from transformers import DetrConfig, AutoModelForObjectDetection
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

class Detr(pl.LightningModule):
    def __init__(self, labels, lr, weight_decay, num_labels, base_model="hustvl/yolos-small"):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained(base_model, 
                                                                 num_labels=len(labels),
                                                                 ignore_mismatched_sizes=True,
                                                                 id2label={str(i): c for i, c in enumerate(labels)},
                                                                 label2id={c: str(i) for i, c in enumerate(labels)},
                                                                 )
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        
        return outputs
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        # labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        labels = batch["labels"]
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        print("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())
        print("validation_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                  weight_decay=self.weight_decay)    
        return optimizer
    
    def save(self):
        self.model.save_pretrained("./models/my_yolos")

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': [{
            "class_labels": torch.tensor(x["labels"]["class_labels"]),            
            "boxes": torch.tensor(x["labels"]["boxes"], dtype=torch.float)
        } for x in batch]
    }


def train_as_torch(model, train_dataset, val_dataset, max_steps=10, model_out_dir="./models"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    print("use: {}".format(accelerator))
    # trainer = Trainer(gpus=0, max_steps=2000, gradient_clip_val=0.1, accumulate_grad_batches=4)
    trainer = Trainer(
        accelerator=accelerator,
        max_steps=max_steps,
        gradient_clip_val=0.1,
        accumulate_grad_batches=4,
        default_root_dir=model_out_dir
        )

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
    
    trainer.fit(model, train_dataloader, val_dataloader)
    model.save()
