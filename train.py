# https://huggingface.co/blog/fine-tune-vit#fromHistory
"""
python train.py

学習データは./dataset/d.jsonを使用
modelは./modelsにsave
"""


from transformers import YolosFeatureExtractor, YolosForObjectDetection, AutoFeatureExtractor
from transformers import TrainingArguments, Trainer

import torch
import numpy as np
from datasets import load_metric, load_dataset
from datasets import Features, Value, ClassLabel, Image as DatasetImage
import os
from PIL import Image
import sys

from train_as_torch import train_as_torch, Detr

_CLASS_NAMES = ["turnip"]
DATASET = "./dataset/d.json"
MODEL_OUTPUT="./model"
BASE_MODEL="hustvl/yolos-small"
    
def load_raw_dataset(ds_path):
    ds = load_dataset("json", data_files=ds_path, split="train")
    ds = ds.train_test_split(test_size=0.3)

    return ds
        

def build_model(model_name):
    # labels = ds['train'].features['labels'].names
    labels = _CLASS_NAMES

    model = YolosForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True
    )
    return model


def get_transform(feature_extractor):
    def transform(example_batch):
        print(example_batch["image_path"])
        example_batch["image"] = [ np.array(Image.open(v)) for v in example_batch["image_path"]]
        # Take a list of PIL images and turn them to pixel values
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        ## inputs = { 'pixel_values': [...] }
        
        # Don't forget to include the labels!
        inputs['labels'] = example_batch['labels']        
        return inputs
    return transform


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def train(feature_extractor, model, prepared_ds):
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=feature_extractor,
        )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def main():
    steps = sys.argv[-1]
    
    model_name = BASE_MODEL    
    feature_extractor = YolosFeatureExtractor.from_pretrained(model_name)

    ds = load_raw_dataset("./dataset/d.json")        
    transform = get_transform(feature_extractor)
    prepared_ds = ds.with_transform(transform)

    # model = build_model(model_name)
    # train(model, feature_extractor, prepared_ds)

    labels = _CLASS_NAMES
    model = Detr(labels, lr=2.5e-5, weight_decay=1e-4, num_labels=1, base_model=model_name)
    train_as_torch(model,
                   prepared_ds["train"],
                   prepared_ds["test"],
                   max_steps=steps,
                   model_out_dir=MODEL_OUTPUT
                   )
    
if __name__ == "__main__":
    main()
