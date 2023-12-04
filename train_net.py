import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from assigner import Assigner
from darknet import DarkNet
from face_dataset import WiderFaceDataset
from loss_functions import DetLoss
from predict import net_predict_bbox

train_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_train_bbx_gt.txt',
                                 root_dir='wider_face_dataset/WIDER_train/images')
val_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_val_bbx_gt.txt',
                               root_dir='wider_face_dataset/WIDER_val/images')

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

anchor_size = 5
net = DarkNet(anchor_size=anchor_size)
os.makedirs('checkpoint', exist_ok=True)
# net.load_state_dict(torch.load('checkpoint/model_e_0_train_loss_0.73_val_loss_2.34.pth'))
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
assigner = Assigner((640, 640), [(20, 20), (10, 10), (5, 5)])
det_loss = DetLoss(assigner.anchors)

for epoch in range(100):
    train_loss = 0
    net.train()
    for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        bs = data['image'].shape[0]
        image = data['image']
        gt_boxes = data['boxes']

        pred_s, pred_m, pred_l = net(image)
        pred = torch.cat([pred_s.reshape(bs, anchor_size, -1),
                          pred_m.reshape(bs, anchor_size, -1),
                          pred_l.reshape(bs, anchor_size, -1)], dim=-1)
        pred = torch.permute(pred, (0, 2, 1))
        pred_bbox_s, pred_prob_s = net_predict_bbox(pred, assigner.anchors_wh, assigner.anchors_x1y1)

        batch_pos_mask, batch_neg_mask, weights = assigner(gt_boxes, image, pred_bboxes=pred_bbox_s)

        cls_loss, reg_loss = det_loss(pred, weights, batch_pos_mask, batch_neg_mask, gt_boxes, image, debug=False)
        loss = cls_loss + reg_loss

        net.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print(f'epoch: {epoch}, step: {step} loss: {loss:.4f}, cls_loss: {cls_loss:.4f}, reg_loss: {reg_loss:.4f}')

    train_loss = train_loss / len(train_dataloader)

    val_loss = 0
    net.eval()
    with torch.no_grad():
        for step, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            bs = data['image'].shape[0]
            image = data['image']
            gt_boxes = data['boxes']

            pred_s, pred_m, pred_l = net(image)
            pred = torch.cat([pred_s.reshape(bs, anchor_size, -1),
                              pred_m.reshape(bs, anchor_size, -1),
                              pred_l.reshape(bs, anchor_size, -1)], dim=-1)
            pred = torch.permute(pred, (0, 2, 1))
            pred_bbox_s, pred_prob_s = net_predict_bbox(pred, assigner.anchors_wh, assigner.anchors_x1y1)

            batch_pos_mask, batch_neg_mask, weights = assigner(gt_boxes, image)

            cls_loss, reg_loss = det_loss(pred, weights, batch_pos_mask, batch_neg_mask, gt_boxes, image, debug=False)
            loss = cls_loss + reg_loss
            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)

    torch.save(net.state_dict(), f'checkpoint/model_e_{epoch}_train_loss_{train_loss:.2f}_val_loss_{val_loss:.2f}.pth')
