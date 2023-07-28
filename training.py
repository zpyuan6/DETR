import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
import os
from dataset.CocoDetection import CocoDetection, collate_fn
from torch.utils.data import DataLoader
from coco_eval import CocoEvaluator
import numpy as np
from tqdm import tqdm
from evaluation import prepare_for_coco_detection


CHECKPOINT = 'facebook/detr-resnet-50'
# DATASET_PATH = "F:\\nematoda\\our_dataset\\COCO"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, dataset_name):
        super().__init__()

        image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
        self.image_processor = image_processor

        train_directory = os.path.join(f"F:\\nematoda\\{dataset_name}\\COCO", "train")
        val_directory = os.path.join(f"F:\\nematoda\\{dataset_name}\\COCO", "valid")

        self.train_dataset = CocoDetection(
        image_directory_path=train_directory, 
        image_processor=image_processor, 
        train=True)

        self.val_dataset = CocoDetection(
            image_directory_path=val_directory, 
            image_processor=image_processor, 
        train=False)

        print("Number of training examples:", len(self.train_dataset))
        print("Number of validation examples:", len(self.val_dataset))
        self.categories = self.train_dataset.coco.cats
        print(self.categories)

        self.t_dataloader = DataLoader(dataset=self.train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True, num_workers=4, prefetch_factor=2*4)
        self.v_dataloader = DataLoader(dataset=self.val_dataset, collate_fn=collate_fn, batch_size=4, num_workers=4, prefetch_factor=2*4)


        id2label = {k: v['name'] for k,v in self.categories.items()}

        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pixel_mask = batch['pixel_mask']
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self.t_dataloader

    def val_dataloader(self):
        return self.v_dataloader

def train(dataset_name):
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, dataset_name=dataset_name)
    MAX_EPOCHS = 100
    # pytorch_lightning >= 2.0.0
    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)
    trainer.fit(model)

    model.model.save_pretrained(dataset_name)


if __name__ == "__main__":

    train("AgriNema")
    train("BBBC010")
    train("Celegans")
    train("Microorganism")


    # batch = next(iter(model.train_dataset))
    # outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    # settings
    
    # evaluator = CocoEvaluator(coco_gt=model.val_dataset.coco, iou_types=["bbox"])
    # print("Running evaluation...")

    # for idx, batch in enumerate(tqdm(model.val_dataloader())):
    #     pixel_values = batch["pixel_values"].to(DEVICE)
    #     pixel_mask = batch["pixel_mask"].to(DEVICE)
    #     labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

    #     with torch.no_grad():
    #         outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    #     orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    #     results = model.image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

    #     predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
    #     predictions = prepare_for_coco_detection(predictions)
    #     evaluator.update(predictions)

    # evaluator.synchronize_between_processes()
    # evaluator.accumulate()
    # evaluator.summarize()

    
