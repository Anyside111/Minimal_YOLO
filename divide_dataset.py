import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.augmentations.geometric.resize import LongestMaxSize
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from torch.utils.data import Dataset

transform = A.Compose([LongestMaxSize(max_size=640, always_apply=True),
                       PadIfNeeded(min_height=640, min_width=640, border_mode=0, always_apply=True)],
                      bbox_params=A.BboxParams(format="pascal_voc", label_fields=['category_id']))


def pad_bbox(bbox, pad_to=1):
    if len(bbox) >= pad_to:
        return bbox[:pad_to]
    return bbox + [(0, 0, 0, 0)] * (pad_to - len(bbox))


class WiderFaceDatasetDivide(Dataset):
    # https://huggingface.co/datasets/wider_face
    def __init__(self, txt_file, root_dir, width_range, transform=transform):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.width_range = width_range
        self.annotations = self._parse_txt_file(txt_file)

    def _parse_txt_file(self, txt_file):
        """
        Parses the txt file and returns a list of annotations.
        """
        annotations = []
        with open(txt_file, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            img_path = lines[i].strip()
            num_faces = int(lines[i + 1].strip())
            if num_faces != 1:
                i += 2 + num_faces
                continue

            boxes = []
            for j in range(num_faces):
                box = list(map(int, lines[i + 2 + j].strip().split()))
                boxes.append(box)
                w = box[2]
                if self.width_range[0] <= w <= self.width_range[1]:
                    boxes.append(box)
            if boxes:
                annotations.append({'img_path': img_path, 'boxes': boxes})
            i += 2 + num_faces

        return annotations




    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations[idx]['img_path'])
        image = Image.open(img_path).convert('RGB')
        boxes = self.annotations[idx]['boxes']
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes if box[2] > 0 and box[3] > 0]
        sample = {'image': image, 'boxes': boxes}

        transformed = self.transform(image=np.array(sample['image']), bboxes=sample['boxes'],
                                     category_id=[1] * len(sample['boxes']))
        sample['image'] = (transformed['image'].transpose(2, 0, 1) / 255.).astype(np.float32)
        sample['boxes'] = torch.tensor(pad_bbox(transformed['bboxes']), dtype=torch.float32)

        return sample



def check_dataset(dataset, width_range, dataset_name):
    """check W """
    print(f"Checking {dataset_name}...")
    for i, sample in enumerate(dataset):
        if i >= 10:
            break
        image = sample['image']
        boxes = sample['boxes']
        np_image = np.array(image).transpose(1, 2, 0)[:, :, ::-1].copy()

        valid_boxes = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            w = x2 - x1
            if width_range[0] <= w <= width_range[1]:
                valid_boxes += 1

        if valid_boxes > 0:
            title = f"{dataset_name} - Sample {i} - Width Range: {width_range}"
            plot_image_with_boxes(np_image, boxes, title)

import cv2
def plot_image_with_boxes(image, boxes, title):
    """show image """
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    cv2.putText(image, title, (10, 30), font, font_scale, font_color, 2, cv2.LINE_AA)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train_dataset_small = WiderFaceDatasetDivide(txt_file='wider_face_dataset/wider_face_train_bbx_gt.txt',
                                           root_dir='wider_face_dataset/WIDER_train/images',
                                           width_range=(0, 100))

    train_dataset_medium = WiderFaceDatasetDivide(txt_file='wider_face_dataset/wider_face_train_bbx_gt.txt',
                                            root_dir='wider_face_dataset/WIDER_train/images',
                                            width_range=(100, 200))

    train_dataset_large = WiderFaceDatasetDivide(txt_file='wider_face_dataset/wider_face_train_bbx_gt.txt',
                                           root_dir='wider_face_dataset/WIDER_train/images',
                                           width_range=(200, 500))
    # check dataset show image
    # for sample in train_dataset_small:
    #     image = sample['image']
    #     boxes = sample['boxes']
    #     np_image = np.array(image).transpose(1, 2, 0)[:, :, ::-1].copy()
    #     for box in boxes:
    #         x1, y1, x2, y2 = box
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         cv2.rectangle(np_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.imshow('image', np_image)
    #     cv2.waitKey(0)

    # check dataset division
    check_dataset(train_dataset_small, (0, 100), "Small Width Dataset")
    check_dataset(train_dataset_medium, (100, 200), "Medium Width Dataset")
    check_dataset(train_dataset_large, (200, 500), "Large Width Dataset")





