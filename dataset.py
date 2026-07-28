import os
import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils import iou_width_height as iou

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            label_dir,
            anchors,
            image_size=416,
            S=[13, 26, 52],
            C=20,
            transform=None,
    ):
        # FIX 1: Correctly assign annotations (was mistakenly overwriting self.anchors)
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.C = C
        self.ignore_iou_thresh = 0.5

        # FIX 2: Properly flatten the list of lists for anchors
        # anchors is expected to be: [[(w,h), (w,h), (w,h)], [(w,h), ...], [(w,h), ...]]
        flat_anchors = [item for sublist in anchors for item in sublist]
        self.anchors = torch.tensor(flat_anchors, dtype=torch.float32).view(-1, 2)

        self.num_anchors = self.anchors.shape[0]
        # FIX: Make this dynamic based on the number of scales (len(self.S))
        self.num_anchors_per_scale = self.num_anchors // len(self.S)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_filename = self.annotations.iloc[index, 0]
        label_filename = self.annotations.iloc[index, 1]

        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, label_filename)

        image = np.array(Image.open(img_path).convert('RGB'))

        # Load bboxes, ndmin=2 ensures it works even if there's only 1 bbox in the file
        bboxes_raw = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)

        # Safeguard: Handle empty label files or malformed single-line reads
        if bboxes_raw.ndim == 1:
            bboxes_raw = bboxes_raw.reshape(1, -1)

        if bboxes_raw.shape[1] < 5:
            bboxes_coords = []
            class_labels = []
        else:
            # Separate coordinates and labels for modern Albumentations
            bboxes_coords = bboxes_raw[:, 1:5].tolist()  # [x, y, w, h]
            class_labels = bboxes_raw[:, 0].tolist()  # [class]

        if self.transform:
            # FIX 5: Pass class_labels separately as required by config.py
            augmentations = self.transform(
                image=image,
                bboxes=bboxes_coords,
                class_labels=class_labels
            )
            image = augmentations['image']

            # Recombine into [x, y, w, h, class] format for YOLO logic
            bboxes = [
                list(b) + [c]
                for b, c in zip(augmentations['bboxes'], augmentations['class_labels'])
            ]
        else:
            # Fallback if no transform is applied
            bboxes = [
                list(b) + [c]
                for b, c in zip(bboxes_coords, class_labels)
            ]

        # FIX 3: Use loop variable 's' instead of the list 'S', and use dynamic num_anchors_per_scale
        targets = [
            torch.zeros((self.num_anchors_per_scale, s, s, 6), dtype=torch.float32)
            for s in self.S
        ]

        for box in bboxes:
            x, y, width, height, class_label = box

            # Calculate IoU with all anchors
            iou_anchors = iou(torch.tensor([width, height], dtype=torch.float32), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True)

            has_anchor = [False] * len(self.S)

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                s = self.S[scale_idx]
                i, j = int(s * y), int(s * x)  # i = row (y), j = col (x)

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    # Assign this anchor to the ground truth
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1.0

                    x_cell, y_cell = s * x - j, s * y - i
                    width_cell, height_cell = width * s, height * s

                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell], dtype=torch.float32
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = float(class_label)

                    # FIX 4: Mark that this scale now has an anchor assigned
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Mark as "ignore" (-1) for objectness loss calculation
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1.0

        return image, tuple(targets)