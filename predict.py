import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from assigner import Assigner
from darknet import DarkNet
from face_dataset import WiderFaceDataset
from assigner import calculate_batch_iou


def net_predict_bbox(network_output, anchor_wh, anchor_x1y1):
    pred_dxdy = network_output[..., :2]
    pred_dwdh = network_output[..., 2:4]
    pred_logit = network_output[..., 4]

    center_offset_scale = F.sigmoid(pred_dxdy)
    hw_scale = torch.exp(pred_dwdh)
    pred_prob = torch.sigmoid(pred_logit)

    pred_bbox_cxy = center_offset_scale * anchor_wh.unsqueeze(0) + anchor_x1y1.unsqueeze(0)
    pred_bbox_wh = hw_scale * anchor_wh.unsqueeze(0)
    pred_bbox = torch.cat([pred_bbox_cxy - pred_bbox_wh / 2, pred_bbox_cxy + pred_bbox_wh / 2], dim=-1)

    return pred_bbox, pred_prob


if __name__ == '__main__':
    import cv2
    from tqdm import tqdm

    debug = False
    test_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_val_bbx_gt.txt',
                                    root_dir='wider_face_dataset/WIDER_val/images')
    # test_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_train_bbx_gt.txt',
    #                                 root_dir='wider_face_dataset/WIDER_train/images')

    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    anchor_size = 5
    net = DarkNet(anchor_size=anchor_size)
    net.load_state_dict(torch.load('checkpoint/model_e_29_train_loss_1.06_val_loss_1.25.pth'))

    assigner = Assigner((640, 640), [(20, 20), (10, 10), (5, 5)])

    mAP = MeanAveragePrecision()

    iou_count = 0
    for step, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        bs = data['image'].shape[0]
        image = data['image']
        gt_boxes = data['boxes']

        with torch.no_grad():
            pred_s, pred_m, pred_l = net(image)
            pred = torch.cat([pred_s.reshape(bs, anchor_size, -1),
                              pred_m.reshape(bs, anchor_size, -1),
                              pred_l.reshape(bs, anchor_size, -1)], dim=-1)
            pred = torch.permute(pred, (0, 2, 1))
            pred_bbox_each_anchor, pred_prob_each_anchor = net_predict_bbox(pred, assigner.anchors_wh,
                                                                            assigner.anchors_x1y1)

            max_idx_p = torch.argmax(pred_prob_each_anchor, dim=1).unsqueeze(-1)
            max_idx_b = max_idx_p.unsqueeze(-1).expand(-1, -1, 4)
            pred_bbox_s = torch.gather(pred_bbox_each_anchor, 1, max_idx_b)
            pred_prob_s = torch.gather(pred_prob_each_anchor, 1, max_idx_p)
            pred_label_s = torch.ones_like(pred_prob_s, dtype=torch.long)

            pred_for_metrics = [dict(zip(["boxes", "scores", "labels"], pred)) for pred in
                                zip(pred_bbox_s, pred_prob_s, pred_label_s)]

            target_for_metrics = [dict(zip(["boxes", "labels"], target)) for target in
                                  zip(gt_boxes, torch.ones_like(pred_label_s))]
            mAP.update(pred_for_metrics, target_for_metrics)

            ious = calculate_batch_iou(gt_boxes, pred_bbox_s)
            iou_count += (ious > 0.5).sum()

        if debug:
            image = image.permute(0, 2, 3, 1).numpy()
            max_anchor_s = torch.gather(assigner.anchors.expand(bs, -1, -1), 1, max_idx_b)
            for b_i in range(bs):
                im = image[b_i][:, :, ::-1].copy()

                pred_bbox = pred_bbox_s[b_i][0]
                pred_prob = pred_prob_s[b_i][0]
                max_anchor = max_anchor_s[b_i][0]
                gt_box = gt_boxes[b_i][0]

                x1, y1, x2, y2 = gt_box
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
                x1, y1, x2, y2 = pred_bbox
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
                cv2.putText(im, f'{pred_prob:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                x1, y1, x2, y2 = max_anchor
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)

                cv2.imshow('image', im)
                cv2.waitKey(0)

    print(iou_count / len(test_dataset))
    print(mAP.compute())
