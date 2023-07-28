from dataset.CocoDetection import CocoDetection, visualize_data,collate_fn
from transformers import DetrForObjectDetection, DetrImageProcessor
from torch.utils.data import DataLoader
import torch
from coco_eval import CocoEvaluator
# from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
import os

MODEL_PATH = "Microorganism"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
DATASET_PATH = f"F:\\nematoda\\{MODEL_PATH}\\COCO"

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


if __name__ == "__main__":
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    model.to(DEVICE)

    val_directory = os.path.join(DATASET_PATH, "valid")
    val_dataset = CocoDetection(
        image_directory_path=val_directory, 
        image_processor=image_processor, 
        train=False)


    val_dataloader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=4)

    # visualize_data(val_dataset)

    # for i, item in enumerate(val_dataloader):
    #     print(i, len(item))

    evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])

    # print("Running evaluation...")

    for idx, batch in enumerate(tqdm(val_dataloader)):
        pixel_values = batch["pixel_values"].to(DEVICE)
        pixel_mask = batch["pixel_mask"].to(DEVICE)
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=orig_target_sizes)

        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        print(predictions)
        evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
