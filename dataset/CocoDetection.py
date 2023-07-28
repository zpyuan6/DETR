import os
import random
import cv2
import supervision as sv
import torchvision
from transformers import DetrImageProcessor
import matplotlib
import matplotlib.pyplot as plt
from tkinter import *


# settings
ANNOTATION_FILE_NAME = "_annotations.coco.json"
# CHECKPOINT = 'facebook/detr-resnet-50'


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible 
    # to directly batch together images. Hence they pad the images to the biggest 
    # resolution in a given batch, and create a corresponding binary pixel_mask 
    # which indicates which pixels are real/which are padding
    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

def visualize_data(coocdataset:CocoDetection):
    image_ids = coocdataset.coco.getImgIds()
    image_id = random.choice(image_ids)
    print('Image #{}'.format(image_id))

    # load image and annotatons 
    image = coocdataset.coco.loadImgs(image_id)[0]
    annotations = coocdataset.coco.imgToAnns[image_id]
    image_path = os.path.join(coocdataset.root, image['file_name'])
    image = cv2.imread(image_path)

    # annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)

    # we will use id2label function for training
    categories = coocdataset.coco.cats
    print(categories)
    id2label = {k: v['name'] for k,v in categories.items()}

    labels = [
        f"{id2label[class_id]}" 
        for _, _, class_id, _ 
        in detections
    ]

    box_annotator = sv.BoxAnnotator()
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    matplotlib.use('TkAgg')
    sv.show_frame_in_notebook(frame, (16, 16))

if __name__ == "__main__":
    dataset_path = "F:\\nematoda\\our_dataset\\COCO"

    image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

    train_directory = os.path.join(dataset_path, "train")
    val_directory = os.path.join(dataset_path, "valid")

    train_dataset = CocoDetection(
        image_directory_path=train_directory, 
        image_processor=image_processor, 
        train=True)
    val_dataset = CocoDetection(
        image_directory_path=val_directory, 
        image_processor=image_processor, 
        train=False)


    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    visualize_data(train_dataset)

