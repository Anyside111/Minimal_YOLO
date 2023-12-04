import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from assigner import Assigner
from darknet import DarkNet
from face_dataset import WiderFaceDataset


def net_predict_bbox(network_output, anchor_wh, anchor_x1y1):
    pred_dxdy = network_output[..., :2]
    pred_dwdh = network_output[..., 2:4]
    pred_logit = network_output[..., 4]

    center_offset_scale = F.sigmoid(pred_dxdy) # bs * 525 anchors *2
    hw_scale = torch.exp(pred_dwdh)
    pred_prob = torch.sigmoid(pred_logit)

    pred_bbox_cxy = center_offset_scale * anchor_wh.unsqueeze(0) + anchor_x1y1.unsqueeze(0)
    pred_bbox_wh = hw_scale * anchor_wh.unsqueeze(0)
    pred_bbox = torch.cat([pred_bbox_cxy - pred_bbox_wh / 2, pred_bbox_cxy + pred_bbox_wh / 2], dim=-1)

    return pred_bbox, pred_prob


if __name__ == '__main__':
    test_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_val_bbx_gt.txt',
                                    root_dir='wider_face_dataset/WIDER_val/images')
    test_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_train_bbx_gt.txt',
                                    root_dir='wider_face_dataset/WIDER_train/images')

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    anchor_size = 5
    net = DarkNet(anchor_size=anchor_size)
    net.load_state_dict(torch.load('checkpoint/model_e_27_train_loss_1.06_val_loss_1.24.pth'))

    assigner = Assigner((640, 640), [(20, 20), (10, 10), (5, 5)])

    for step, data in enumerate(test_dataloader):
        bs = data['image'].shape[0]
        image = data['image']
        gt_boxes = data['boxes']

        pred_s, pred_m, pred_l = net(image)
        pred = torch.cat([pred_s.reshape(bs, anchor_size, -1),
                          pred_m.reshape(bs, anchor_size, -1),
                          pred_l.reshape(bs, anchor_size, -1)], dim=-1)
        pred = torch.permute(pred, (0, 2, 1))
        pred_bbox_s, pred_prob_s = net_predict_bbox(pred, assigner.anchors_wh, assigner.anchors_x1y1)

        image = image.permute(0, 2, 3, 1).numpy()
        for b_i in range(bs):
            im = image[b_i][:, :, ::-1].copy()
            pred_bbox = pred_bbox_s[b_i]
            pred_prob = pred_prob_s[b_i]

            pos_pred_idx = pred_prob > 0.2
            pred_bbox_filter = pred_bbox[pos_pred_idx]
            pred_prob_filter = pred_prob[pos_pred_idx]
            anchor_filter = assigner.anchors[pos_pred_idx]

            sorted_idx = torch.argsort(pred_prob_filter, descending=True)
            sorted_bbox_pred = pred_bbox_filter[sorted_idx]
            sorted_prob_pred = pred_prob_filter[sorted_idx]
            sorted_anchor = anchor_filter[sorted_idx]

            gt_box = gt_boxes[b_i]
            x1, y1, x2, y2 = gt_box[0]
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)

            for anch, pred_bbox, prob in zip(sorted_anchor, sorted_bbox_pred, sorted_prob_pred):
                x1, y1, x2, y2 = pred_bbox
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
                cv2.putText(im, f'{prob:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                x1, y1, x2, y2 = anch
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)

                cv2.imshow('image', im)
                cv2.waitKey(0)

                break
