import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from assigner import Assigner
from assigner import calculate_batch_iou
from darknet import DarkNet
from face_dataset import WiderFaceDataset
from predict import net_predict_bbox
from ciou_loss import Ciou_Loss

train_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_train_bbx_gt.txt',
                                 root_dir='wider_face_dataset/WIDER_train/images')
val_dataset = WiderFaceDataset(txt_file='wider_face_dataset/wider_face_val_bbx_gt.txt',
                               root_dir='wider_face_dataset/WIDER_val/images')

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)


anchor_size = 5
net = DarkNet(anchor_size=anchor_size)
os.makedirs('test_ciou_all', exist_ok=True)
#net.load_state_dict(torch.load('checkpoint/model_e_48_train_loss_0.80_val_loss_0.90_accuracy_0.70.pth'))
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
assigner = Assigner((640, 640), [(20, 20), (10, 10), (5, 5)])
ciou_loss = Ciou_Loss(assigner.anchors)

epoch_arr=[1,2]
train_loss_arr=[3,4]
validation_loss_arr=[5,6]
validation_accuracy_arr=[7,8]

for epoch in range(1,31):
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

        cls_loss, reg_loss = ciou_loss(pred, weights, batch_pos_mask, batch_neg_mask, gt_boxes, image, debug=False)
        loss = cls_loss + reg_loss
        net.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print(f'epoch: {epoch}, step: {step} loss: {loss:.4f}, cls_loss: {cls_loss:.4f}, reg_loss: {reg_loss:.4f}')

    train_loss = train_loss / len(train_dataloader)

    val_loss = 0
    net.eval()
    accuracies = []
    iou_count = 0
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
            pred_bbox_each_anchor, pred_prob_each_anchor = net_predict_bbox(pred, assigner.anchors_wh,
                                                                            assigner.anchors_x1y1)

            max_idx_p = torch.argmax(pred_prob_each_anchor, dim=1).unsqueeze(-1)
            max_idx_b = max_idx_p.unsqueeze(-1).expand(-1, -1, 4)
            pred_bbox_s = torch.gather(pred_bbox_each_anchor, 1, max_idx_b)
            pred_prob_s = torch.gather(pred_prob_each_anchor, 1, max_idx_p)
            pred_label_s = torch.ones_like(pred_prob_s, dtype=torch.long)
            ious = calculate_batch_iou(gt_boxes, pred_bbox_s)
            iou_count += (ious > 0.5).sum()

            batch_pos_mask, batch_neg_mask, weights = assigner(gt_boxes, image)

            cls_loss, reg_loss = ciou_loss(pred, weights, batch_pos_mask, batch_neg_mask, gt_boxes, image, debug=False)
            loss = cls_loss + reg_loss
            val_loss += loss.item()

        val_loss = val_loss / len(val_dataloader)
        accuracy = iou_count / len(val_dataset) #validate on medium dataset
        accuracies.append(accuracy)
        print(f'valid: epoch: {epoch}, val_loss: {val_loss:.4f}, accuracy: {accuracy:.4f}')
    torch.save(net.state_dict(), f'test_ciou_all/a_ciou_e_{epoch}_train_loss_{train_loss:.5f}_val_loss_{val_loss:.5f}_accuracy_{accuracy:.5f}.pth')
    epoch_arr.append(epoch)
    train_loss_arr.append(train_loss)
    validation_loss_arr.append(val_loss)
    validation_accuracy_arr.append(accuracy)


# Specify the file path
file_path = 'test_ciou_all/stat_data.txt'
print(f'epoch_list={epoch_arr},train_loss={train_loss_arr},validation_loss={validation_loss_arr},validation_accuracy={validation_accuracy_arr}')

# Write arrays to the file
with open(file_path, 'w') as file:

    file.write("epoch:\n")
    np.savetxt(file, [epoch_arr], fmt='%d', delimiter=',', header='[', footer=']', comments='')

    file.write("train loss:\n")
    np.savetxt(file, [train_loss_arr], fmt='%.5f', delimiter=',', header='[', footer=']', comments='')

    file.write("\nvalid loss:\n")
    np.savetxt(file, [validation_loss_arr], fmt='%.5f', delimiter=',', header='[', footer=']', comments='')

    file.write("\nvalid accuracy:\n")
    np.savetxt(file, [validation_accuracy_arr], fmt='%.5f', delimiter=',', header='[', footer=']', comments='')

print(f"Arrays have been written to {file_path}")

