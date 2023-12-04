import torch
from torch import nn


def bbox_center(boxes):
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def calculate_batch_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Calculate iou of box1 and box2 in batch. Bboxes are expected to be in x1y1x2y2 format.

    :param box1: box with the shape [N, M1, 4]
    :param box2: box with the shape [N, M2, 4]
    :return iou: iou between box1 and box2 with the shape [N, M1, M2]

    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


class Assigner(nn.Module):
    def __init__(self, im_hw, anchor_num_per_level, topk=9):
        super().__init__()
        self.topk = topk
        self.im_h, self.im_w = im_hw
        anchor_l1_m, anchor_l1_n = anchor_num_per_level[0]
        anchor_l2_m, anchor_l2_n = anchor_num_per_level[1]
        anchor_l3_m, anchor_l3_n = anchor_num_per_level[2]

        l1_unit_h = self.im_h / anchor_l1_m
        l1_unit_w = self.im_w / anchor_l1_n
        l2_unit_h = self.im_h / anchor_l2_m
        l2_unit_w = self.im_w / anchor_l2_n
        l3_unit_h = self.im_h / anchor_l3_m
        l3_unit_w = self.im_w / anchor_l3_n

        # xyxy
        anchors_s = [[l1_unit_w * i, l1_unit_h * j, l1_unit_w * (i + 1), l1_unit_h * (j + 1)]
                     for j in range(anchor_l1_n) for i in range(anchor_l1_m)]
        anchors_m = [[l2_unit_w * i, l2_unit_h * j, l2_unit_w * (i + 1), l2_unit_h * (j + 1)]
                     for j in range(anchor_l2_n) for i in range(anchor_l2_m)]
        anchors_l = [[l3_unit_w * i, l3_unit_h * j, l3_unit_w * (i + 1), l3_unit_h * (j + 1)]
                     for j in range(anchor_l3_n) for i in range(anchor_l3_m)]

        self.anchors = torch.tensor(anchors_s + anchors_m + anchors_l) # concactate anchors
        self.anchors_wh = self.anchors[:, 2:4] - self.anchors[:, :2]
        self.anchors_x1y1 = self.anchors[:, :2]

    def forward(self, gt_bboxes, images, pred_bboxes=None, debug=False):
        bs = gt_bboxes.shape[0]

        batch_gt_anchor_iou = calculate_batch_iou(gt_bboxes, self.anchors.unsqueeze(0))
        batch_positive_mask = torch.zeros_like(batch_gt_anchor_iou, dtype=torch.bool)
        batch_negative_mask = torch.zeros_like(batch_gt_anchor_iou, dtype=torch.bool)

        for i in range(bs):
            gt_anchor_iou = batch_gt_anchor_iou[i]

            gt_centers = bbox_center(gt_bboxes[i])
            anchor_centers = bbox_center(self.anchors)

            gt2anchor_distances = torch.norm(gt_centers.unsqueeze(1) - anchor_centers.unsqueeze(0), dim=-1)
            topk_idxs = torch.topk(gt2anchor_distances, k=self.topk, dim=1, largest=False).indices

            topk_iou = torch.gather(gt_anchor_iou.flatten(end_dim=-2), dim=1, index=topk_idxs.flatten(end_dim=-2))
            topk_iou = topk_iou[topk_iou > 0]
            iou_threshold = topk_iou.mean() + topk_iou.std()

            batch_positive_mask[i] = gt_anchor_iou > iou_threshold
            batch_negative_mask[i] = gt_anchor_iou < iou_threshold

        if debug:
            import cv2
            images = images.permute(0, 2, 3, 1).numpy()
            for i in range(bs):
                positive_anchor = self.anchors[batch_positive_mask[i][0]]
                im = images[i][:, :, ::-1].copy()
                for j in range(len(positive_anchor)):
                    x1, y1, x2, y2 = positive_anchor[j]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 1)
                for j in range(len(gt_bboxes[i])):
                    x1, y1, x2, y2 = gt_bboxes[i][j]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.imshow('image', im)
                cv2.waitKey(0)

        assigned_scores = torch.ones_like(batch_gt_anchor_iou, dtype=torch.float)[:, 0, :]
        if pred_bboxes is not None:
            ious = calculate_batch_iou(gt_bboxes, self.anchors.unsqueeze(0)) * batch_positive_mask
            # ious = calculate_batch_iou(gt_bboxes, pred_bboxes) * batch_positive_mask
            ious = ious[:, 0, :]
            uniform_iou = ious / ious.max(dim=1).values.unsqueeze(-1)
            assigned_scores *= uniform_iou

        return batch_positive_mask, batch_negative_mask, assigned_scores


if __name__ == '__main__':
    Assigner((640, 640), [(80, 80), (40, 40), (20, 20)])
    # Assigner((640, 640), [(20, 20), (10, 10), (5, 5)])
