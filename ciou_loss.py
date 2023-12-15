import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import generalized_box_iou_loss,complete_box_iou_loss


class Ciou_Loss(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

    def forward(self, pred_bbox_cls_s, bbox_weights, pos_idxes, neg_idxes, gt_bboxes, images, debug=False):
        bs = pred_bbox_cls_s.shape[0]
        cls_loss = 0
        reg_loss = 0
        for i in range(bs):
            bbox_weight = bbox_weights[i]
            pred_bbox_cls = pred_bbox_cls_s[i]
            pos_idx = pos_idxes[i][0] # bs * 1 * 525 anchors
            neg_idx = neg_idxes[i][0]
            gt_bbox = gt_bboxes[i][0]

            pos_pred = pred_bbox_cls[pos_idx]
            pos_bbox_w = bbox_weight[pos_idx]
            neg_pred = pred_bbox_cls[neg_idx]

            # cls loss
            if len(pos_pred) > 0:
                positive_loss = -torch.nn.functional.logsigmoid(pos_pred[:, 4]) * pos_bbox_w # pos_bbox_w is gt_anchor_iou
                cls_loss += torch.mean(positive_loss)
            if len(neg_pred) > 0:
                negative_loss = -torch.nn.functional.logsigmoid(-neg_pred[:, 4]) # sigmoid(-x) = 1 - sigmoid(x)
                cls_loss += torch.mean(negative_loss)

            # reg loss
            if len(pos_pred) > 0:
                pos_anchor = self.anchors[pos_idx]
                anchor_wh = pos_anchor[:, 2:4] - pos_anchor[:, :2]
                anchor_x1y1 = pos_anchor[:, :2]

                center_offset_scale = F.sigmoid(pos_pred[:, :2])
                hw_scale = torch.exp(pos_pred[:, 2:4])

                bbox_cxy = center_offset_scale * anchor_wh + anchor_x1y1
                bbox_wh = hw_scale * anchor_wh
                bbox_x1y1x2y2 = torch.cat([bbox_cxy - bbox_wh / 2, bbox_cxy + bbox_wh / 2], dim=-1) # pred bbox
                reg_loss_ = complete_box_iou_loss(bbox_x1y1x2y2, gt_bbox) * pos_bbox_w # use iou loss of pred bbox and gt bbox as weight
                reg_loss += torch.mean(reg_loss_)

                if debug:
                    import cv2

                    image = images[i].permute(1, 2, 0).numpy()[:, :, ::-1].copy()
                    for j in range(len(bbox_x1y1x2y2)):
                        xc, yc = bbox_cxy[j]
                        cv2.circle(image, (int(xc), int(yc)), 3, (0, 255, 0), 2)
                        x1, y1, x2, y2 = bbox_x1y1x2y2[j]
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

                    for j in range(len(pos_anchor)):
                        x1, y1 = anchor_x1y1[j]
                        cv2.circle(image, (int(x1), int(y1)), 1, (0, 0, 0), 2)
                        x1, y1, x2, y2 = pos_anchor[j]
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)

                    x1, y1, x2, y2 = gt_bbox
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
                    cv2.imshow('image', image)
                    cv2.waitKey(1)
                    print()
                    if torch.min(hw_scale).item() < 1:
                        print()

        cls_loss = cls_loss / bs
        reg_loss = reg_loss / bs

        return cls_loss, reg_loss

def complete_iou_loss(pred_boxes, target_boxes, anchor_wh):
    # Ensure target_boxes has two dimensions
    if target_boxes.dim() == 1:
        target_boxes = target_boxes.unsqueeze(0)

    x1, y1 = torch.split(target_boxes[:, :1], 1, dim=1)
    x2, y2 = torch.split(target_boxes[:, 1:], 1, dim=1)

    pred_x1, pred_y1 = torch.split(pred_boxes[:, :1], 1, dim=1)
    pred_x2, pred_y2 = torch.split(pred_boxes[:, 1:], 1, dim=1)

    target_cx, target_cy = (x1 + x2) / 2, (y1 + y2) / 2
    target_w, target_h = x2 - x1, y2 - y1

    pred_cx, pred_cy = (pred_x1 + pred_x2) / 2, (pred_y1 + pred_y2) / 2
    pred_w, pred_h = pred_x2 - pred_x1, pred_y2 - pred_y1

    iou = bbox_iou(pred_boxes, target_boxes)

    v = 4 / (math.pi ** 2) * torch.pow(torch.atan(anchor_wh[:, 0] / anchor_wh[:, 1]) - torch.atan(target_w / target_h),
                                       2)

    alpha = v / (1 - iou + v)

    ciou = iou - (torch.pow(pred_cx - target_cx, 2) + torch.pow(pred_cy - target_cy, 2)) / torch.pow(pred_w + target_w,
                                                                                                     2) - alpha * v

    ciou_loss = 1 - ciou
    return ciou_loss

def bbox_iou(box1, box2):
    intersect_wh = torch.clamp((torch.min(box1[:, 2:], box2[:, 2:]) - torch.max(box1[:, :2], box2[:, :2])).clamp(min=0), min=0)
    intersect = intersect_wh[:, 0] * intersect_wh[:, 1]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1 + area2 - intersect

    iou = intersect / union
    return iou