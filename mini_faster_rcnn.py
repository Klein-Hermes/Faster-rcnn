"""
Mini Faster R-CNN — 单文件极简实现
=============================================
论文: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
      Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun  (NeurIPS 2015)

核心流水线:
    图片 → CNN Backbone → Anchor 生成 → RPN → Proposal (NMS) → ROI Pooling → 分类 + 回归

本文件 ≈ 500 行，保留了 Faster R-CNN 论文的全部核心模块，
去掉了工程细节（FPN, 多尺度训练, 分布式等），适合学习与教学。

依赖: torch >= 1.10, torchvision >= 0.11
运行: python mini_faster_rcnn.py          # 用合成数据训练 + 推理 demo
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms, box_iou, roi_pool


# ============================================================
#  1. CNN Backbone  —— 简化版 VGG 风格特征提取器
# ============================================================
class SimpleBackbone(nn.Module):
    """
    轻量 CNN 骨干网络，将输入图片下采样 16 倍并提取特征。
    论文中使用 VGG-16 / ZF-Net，这里用 4 组 Conv+BN+ReLU+Pool 替代。
    输入: (B, 3, H, W)  →  输出: (B, 256, H/16, W/16)
    """

    def __init__(self, out_channels=256):
        super().__init__()
        layers = []
        in_ch = 3
        # 4 次下采样，每次 stride=2 的 MaxPool → 总共 ÷16
        for out_ch in [64, 128, 256, out_channels]:
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        return self.features(x)


# ============================================================
#  2. Anchor 生成器
# ============================================================
class AnchorGenerator:
    """
    在特征图的每个空间位置生成 k 个 anchor boxes。
    论文使用 3 种尺度 × 3 种长宽比 = 9 个 anchor / 位置。

    anchor 格式: (x1, y1, x2, y2)，坐标对应原图。
    """

    def __init__(self, sizes=(64, 128, 256), ratios=(0.5, 1.0, 2.0), stride=16):
        self.sizes = sizes
        self.ratios = ratios
        self.stride = stride
        # 预计算 base anchors（以 (0,0) 为中心）
        self.base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self):
        """生成 k 个以原点为中心的 base anchor。"""
        anchors = []
        for s in self.sizes:
            area = s * s
            for r in self.ratios:
                # w * h = area,  w / h = r  →  w = sqrt(area * r)
                w = math.sqrt(area * r)
                h = area / w
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def __call__(self, feature_map_size, image_size, device):
        """
        Args:
            feature_map_size: (feat_H, feat_W)
            image_size:       (img_H, img_W)
        Returns:
            anchors: (feat_H * feat_W * k, 4)   原图坐标
        """
        feat_h, feat_w = feature_map_size
        # 在特征图网格上平移 base anchors
        shift_x = torch.arange(0, feat_w, device=device) * self.stride + self.stride // 2
        shift_y = torch.arange(0, feat_h, device=device) * self.stride + self.stride // 2
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1).reshape(-1, 4).float()

        base = self.base_anchors.to(device)
        # shifts: (N, 1, 4) + base: (1, k, 4) → (N, k, 4) → (N*k, 4)
        all_anchors = (shifts.unsqueeze(1) + base.unsqueeze(0)).reshape(-1, 4)

        # 裁剪到图片范围
        img_h, img_w = image_size
        all_anchors[:, 0::2].clamp_(0, img_w)
        all_anchors[:, 1::2].clamp_(0, img_h)
        return all_anchors

    @property
    def num_anchors_per_location(self):
        return len(self.sizes) * len(self.ratios)


# ============================================================
#  3. RPN (Region Proposal Network)
# ============================================================
class RPN(nn.Module):
    """
    论文核心创新：用一个小型全卷积网络在特征图上滑动，
    同时预测每个 anchor 的 objectness 得分和边框回归偏移量。

    结构:  3×3 conv → 1×1 conv (cls)  +  1×1 conv (reg)
    """

    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_score = nn.Conv2d(in_channels, num_anchors, 1)   # 二分类: 前景 / 背景
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)  # 4 个回归偏移

        # 初始化
        for layer in [self.conv, self.cls_score, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feature_map):
        """
        Returns:
            rpn_cls:  (B, k, H, W)    objectness logits
            rpn_reg:  (B, k*4, H, W)  bbox deltas
        """
        x = F.relu(self.conv(feature_map), inplace=True)
        rpn_cls = self.cls_score(x)    # (B, k, H, W)
        rpn_reg = self.bbox_pred(x)    # (B, k*4, H, W)
        return rpn_cls, rpn_reg


# ============================================================
#  4. Proposal 生成 (解码 + NMS)
# ============================================================
def decode_boxes(anchors, deltas):
    """
    将 RPN 预测的 (dx, dy, dw, dh) 偏移量 + anchor 解码为实际框坐标。
    论文公式:
        x = dx * w_a + x_a,   y = dy * h_a + y_a
        w = exp(dw) * w_a,    h = exp(dh) * h_a
    """
    # anchor 中心和宽高
    w_a = anchors[:, 2] - anchors[:, 0]
    h_a = anchors[:, 3] - anchors[:, 1]
    cx_a = anchors[:, 0] + 0.5 * w_a
    cy_a = anchors[:, 1] + 0.5 * h_a

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    dw = dw.clamp(max=4.0)  # 防止 exp 爆炸
    dh = dh.clamp(max=4.0)

    cx = dx * w_a + cx_a
    cy = dy * h_a + cy_a
    w = torch.exp(dw) * w_a
    h = torch.exp(dh) * h_a

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)


def encode_boxes(anchors, gt_boxes):
    """将 ground truth 框编码为相对 anchor 的偏移量 (dx, dy, dw, dh)。"""
    w_a = (anchors[:, 2] - anchors[:, 0]).clamp(min=1)
    h_a = (anchors[:, 3] - anchors[:, 1]).clamp(min=1)
    cx_a = anchors[:, 0] + 0.5 * w_a
    cy_a = anchors[:, 1] + 0.5 * h_a

    w_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1)
    h_gt = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1)
    cx_gt = gt_boxes[:, 0] + 0.5 * w_gt
    cy_gt = gt_boxes[:, 1] + 0.5 * h_gt

    dx = (cx_gt - cx_a) / w_a
    dy = (cy_gt - cy_a) / h_a
    dw = torch.log(w_gt / w_a)
    dh = torch.log(h_gt / h_a)
    return torch.stack([dx, dy, dw, dh], dim=1)


def generate_proposals(rpn_cls, rpn_reg, anchors, image_size,
                        pre_nms_top_n=6000, post_nms_top_n=300,
                        nms_thresh=0.7, min_size=8):
    """
    从 RPN 输出生成 proposals：
      1) 解码框坐标   2) 裁剪到图片   3) 过滤小框   4) NMS
    """
    device = rpn_cls.device
    scores = rpn_cls.sigmoid().reshape(-1)
    deltas = rpn_reg.reshape(-1, 4)

    # 取 top-N 得分
    if pre_nms_top_n > 0 and scores.shape[0] > pre_nms_top_n:
        _, top_idx = scores.topk(pre_nms_top_n)
        scores = scores[top_idx]
        deltas = deltas[top_idx]
        anchors = anchors[top_idx]

    # 解码
    proposals = decode_boxes(anchors, deltas)

    # 裁剪到图片范围
    img_h, img_w = image_size
    proposals[:, 0::2].clamp_(0, img_w)
    proposals[:, 1::2].clamp_(0, img_h)

    # 过滤过小的框
    ws = proposals[:, 2] - proposals[:, 0]
    hs = proposals[:, 3] - proposals[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    proposals, scores = proposals[keep], scores[keep]

    # NMS
    keep = nms(proposals, scores, nms_thresh)
    if post_nms_top_n > 0:
        keep = keep[:post_nms_top_n]
    return proposals[keep], scores[keep]


# ============================================================
#  5. ROI Pooling + 分类/回归 Head
# ============================================================
class RCNNHead(nn.Module):
    """
    Fast R-CNN 的检测头：
      ROI Pooling → FC → FC → cls_score + bbox_pred

    论文中使用 7×7 的 ROI Pooling 输出，再接两层 FC。
    """

    def __init__(self, in_channels, roi_size, num_classes):
        super().__init__()
        self.roi_size = roi_size
        self.num_classes = num_classes

        fc_in = in_channels * roi_size * roi_size
        self.fc1 = nn.Linear(fc_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.cls_score = nn.Linear(256, num_classes)       # 含背景类
        self.bbox_pred = nn.Linear(256, num_classes * 4)   # 每类一套回归参数

    def forward(self, feature_map, rois):
        """
        Args:
            feature_map: (B, C, H, W)  骨干网络输出
            rois:        (N, 5)  每行 [batch_idx, x1, y1, x2, y2]
        Returns:
            cls_logits:  (N, num_classes)
            bbox_deltas: (N, num_classes * 4)
        """
        # spatial_scale = 特征图尺寸 / 原图尺寸 = 1/16
        pooled = roi_pool(feature_map, rois, output_size=self.roi_size,
                          spatial_scale=1.0 / 16.0)  # (N, C, roi_size, roi_size)
        x = pooled.flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return cls_logits, bbox_deltas


# ============================================================
#  6. 完整 Faster R-CNN
# ============================================================
class MiniFasterRCNN(nn.Module):
    """
    将上述模块组合为完整的 Faster R-CNN 检测器。

    训练时计算:  RPN loss (cls + reg)  +  RCNN loss (cls + reg)
    推理时输出:  检测框、类别、得分
    """

    def __init__(self, num_classes, backbone_channels=256, roi_size=7):
        super().__init__()
        self.num_classes = num_classes  # 含背景 (class 0 = bg)

        # --- 模块组装 ---
        self.backbone = SimpleBackbone(out_channels=backbone_channels)
        self.anchor_gen = AnchorGenerator(sizes=(64, 128, 256),
                                          ratios=(0.5, 1.0, 2.0),
                                          stride=16)
        k = self.anchor_gen.num_anchors_per_location
        self.rpn = RPN(backbone_channels, k)
        self.head = RCNNHead(backbone_channels, roi_size, num_classes)

        # --- 训练超参 ---
        self.rpn_pos_iou = 0.7       # anchor 正样本 IoU 阈值
        self.rpn_neg_iou = 0.3       # anchor 负样本 IoU 阈值
        self.rpn_batch_size = 256     # RPN 采样数
        self.rpn_pos_frac = 0.5      # 正样本比例

        self.rcnn_pos_iou = 0.5      # proposal 正样本阈值
        self.rcnn_batch_size = 64     # RCNN 采样数
        self.rcnn_pos_frac = 0.25

    # -------------------- 训练逻辑 --------------------

    def _sample_rpn_targets(self, anchors, gt_boxes):
        """
        为 RPN 采样正/负 anchor，返回 labels 和 回归 targets。
        正样本: IoU ≥ 0.7 或与某个 GT 有最高 IoU 的 anchor
        负样本: IoU < 0.3
        """
        iou = box_iou(anchors, gt_boxes)  # (A, G)
        max_iou, matched_gt_idx = iou.max(dim=1)

        labels = torch.full((anchors.shape[0],), -1, dtype=torch.float32,
                            device=anchors.device)
        labels[max_iou < self.rpn_neg_iou] = 0
        labels[max_iou >= self.rpn_pos_iou] = 1
        # 保证每个 GT 至少有一个正样本
        best_anchor_per_gt = iou.argmax(dim=0)
        labels[best_anchor_per_gt] = 1

        # 子采样：控制正负样本比例
        pos_idx = (labels == 1).nonzero(as_tuple=False).squeeze(1)
        neg_idx = (labels == 0).nonzero(as_tuple=False).squeeze(1)
        n_pos = min(int(self.rpn_batch_size * self.rpn_pos_frac), len(pos_idx))
        n_neg = min(self.rpn_batch_size - n_pos, len(neg_idx))
        if len(pos_idx) > n_pos:
            pos_idx = pos_idx[torch.randperm(len(pos_idx), device=anchors.device)[:n_pos]]
        if len(neg_idx) > n_neg:
            neg_idx = neg_idx[torch.randperm(len(neg_idx), device=anchors.device)[:n_neg]]

        sampled = torch.cat([pos_idx, neg_idx])
        sampled_labels = labels[sampled]

        # 回归 targets (仅正样本有意义)
        matched_gt = gt_boxes[matched_gt_idx[sampled]]
        reg_targets = encode_boxes(anchors[sampled], matched_gt)

        return sampled, sampled_labels, reg_targets

    def _sample_rcnn_targets(self, proposals, gt_boxes, gt_labels):
        """
        为第二阶段采样正/负 proposals。
        正样本: IoU ≥ 0.5 → 分配 GT 类别
        负样本: IoU < 0.5 → 标签 = 0 (背景)
        """
        iou = box_iou(proposals, gt_boxes)
        max_iou, matched_gt_idx = iou.max(dim=1)

        labels = gt_labels[matched_gt_idx]
        labels[max_iou < self.rcnn_pos_iou] = 0  # 背景

        pos_idx = (labels > 0).nonzero(as_tuple=False).squeeze(1)
        neg_idx = (labels == 0).nonzero(as_tuple=False).squeeze(1)
        n_pos = min(int(self.rcnn_batch_size * self.rcnn_pos_frac), len(pos_idx))
        n_neg = min(self.rcnn_batch_size - n_pos, len(neg_idx))
        if len(pos_idx) > n_pos:
            pos_idx = pos_idx[torch.randperm(len(pos_idx), device=proposals.device)[:n_pos]]
        if len(neg_idx) > n_neg:
            neg_idx = neg_idx[torch.randperm(len(neg_idx), device=proposals.device)[:n_neg]]

        sampled = torch.cat([pos_idx, neg_idx])
        sampled_labels = labels[sampled]
        sampled_proposals = proposals[sampled]

        # 回归 targets
        matched_gt = gt_boxes[matched_gt_idx[sampled]]
        reg_targets = encode_boxes(sampled_proposals, matched_gt)

        return sampled_proposals, sampled_labels, reg_targets

    def forward(self, images, targets=None):
        """
        Args:
            images:  (B, 3, H, W) 输入图片 tensor
            targets: list[dict] 每张图的标注, 含 'boxes' (N,4) 和 'labels' (N,)
                     训练时必须提供; 推理时为 None

        Returns:
            训练: dict  {'rpn_cls_loss', 'rpn_reg_loss', 'rcnn_cls_loss', 'rcnn_reg_loss'}
            推理: list[dict]  每张图一个 dict, 含 'boxes', 'labels', 'scores'
        """
        B, _, img_h, img_w = images.shape
        device = images.device
        image_size = (img_h, img_w)

        # ---- Step 1: CNN Backbone ----
        feat = self.backbone(images)                # (B, C, fH, fW)
        feat_h, feat_w = feat.shape[2], feat.shape[3]

        # ---- Step 2: 生成 Anchors ----
        anchors = self.anchor_gen((feat_h, feat_w), image_size, device)  # (A, 4)
        k = self.anchor_gen.num_anchors_per_location

        # ---- Step 3: RPN 前向 ----
        rpn_cls, rpn_reg = self.rpn(feat)           # (B, k, fH, fW), (B, k*4, fH, fW)

        if self.training:
            return self._forward_train(feat, rpn_cls, rpn_reg, anchors,
                                       image_size, targets, k, B)
        else:
            return self._forward_infer(feat, rpn_cls, rpn_reg, anchors,
                                       image_size, B)

    def _forward_train(self, feat, rpn_cls, rpn_reg, anchors,
                       image_size, targets, k, B):
        total_rpn_cls_loss = 0
        total_rpn_reg_loss = 0
        total_rcnn_cls_loss = 0
        total_rcnn_reg_loss = 0

        for b in range(B):
            gt_boxes = targets[b]['boxes']     # (G, 4)
            gt_labels = targets[b]['labels']   # (G,)

            # --- RPN Loss ---
            cls_b = rpn_cls[b].permute(1, 2, 0).reshape(-1)        # (A,)
            reg_b = rpn_reg[b].permute(1, 2, 0).reshape(-1, 4)     # (A, 4)

            sampled_idx, sampled_labels, rpn_reg_targets = \
                self._sample_rpn_targets(anchors, gt_boxes)

            rpn_cls_loss = F.binary_cross_entropy_with_logits(
                cls_b[sampled_idx], sampled_labels)
            pos_mask = sampled_labels == 1
            if pos_mask.sum() > 0:
                rpn_reg_loss = F.smooth_l1_loss(
                    reg_b[sampled_idx][pos_mask],
                    rpn_reg_targets[pos_mask], beta=1.0)
            else:
                rpn_reg_loss = torch.tensor(0.0, device=feat.device)

            total_rpn_cls_loss += rpn_cls_loss
            total_rpn_reg_loss += rpn_reg_loss

            # --- Step 4: 生成 Proposals ---
            with torch.no_grad():
                proposals, _ = generate_proposals(
                    rpn_cls[b], rpn_reg[b], anchors, image_size,
                    pre_nms_top_n=2000, post_nms_top_n=256)

            if proposals.shape[0] == 0:
                continue

            # --- Step 5 & 6: ROI Pooling + 分类/回归 Head ---
            sampled_props, rcnn_labels, rcnn_reg_targets = \
                self._sample_rcnn_targets(proposals, gt_boxes, gt_labels)

            # 构造 rois: (N, 5)  [batch_idx, x1, y1, x2, y2]
            batch_idx_col = torch.full((sampled_props.shape[0], 1), b,
                                       dtype=torch.float32, device=feat.device)
            rois = torch.cat([batch_idx_col, sampled_props], dim=1)

            cls_logits, bbox_deltas = self.head(feat, rois)

            # RCNN 分类 loss
            rcnn_cls_loss = F.cross_entropy(cls_logits, rcnn_labels.long())

            # RCNN 回归 loss (仅正样本，取对应类别的回归参数)
            fg_mask = rcnn_labels > 0
            if fg_mask.sum() > 0:
                fg_labels = rcnn_labels[fg_mask].long()
                fg_deltas = bbox_deltas[fg_mask]
                # 取出每个正样本对应类别的 4 个回归值
                idx = torch.arange(fg_labels.shape[0], device=feat.device)
                fg_pred = fg_deltas.reshape(-1, self.num_classes, 4)[idx, fg_labels]
                fg_targets = rcnn_reg_targets[fg_mask]
                rcnn_reg_loss = F.smooth_l1_loss(fg_pred, fg_targets, beta=1.0)
            else:
                rcnn_reg_loss = torch.tensor(0.0, device=feat.device)

            total_rcnn_cls_loss += rcnn_cls_loss
            total_rcnn_reg_loss += rcnn_reg_loss

        return {
            'rpn_cls_loss': total_rpn_cls_loss / B,
            'rpn_reg_loss': total_rpn_reg_loss / B,
            'rcnn_cls_loss': total_rcnn_cls_loss / B,
            'rcnn_reg_loss': total_rcnn_reg_loss / B,
        }

    @torch.no_grad()
    def _forward_infer(self, feat, rpn_cls, rpn_reg, anchors, image_size, B):
        results = []
        for b in range(B):
            proposals, _ = generate_proposals(
                rpn_cls[b], rpn_reg[b], anchors, image_size,
                pre_nms_top_n=6000, post_nms_top_n=300)

            if proposals.shape[0] == 0:
                results.append({'boxes': torch.empty(0, 4), 'labels': torch.empty(0),
                                'scores': torch.empty(0)})
                continue

            batch_idx_col = torch.full((proposals.shape[0], 1), b,
                                       dtype=torch.float32, device=feat.device)
            rois = torch.cat([batch_idx_col, proposals], dim=1)
            cls_logits, bbox_deltas = self.head(feat, rois)

            probs = F.softmax(cls_logits, dim=1)           # (N, num_classes)

            all_boxes, all_labels, all_scores = [], [], []
            for c in range(1, self.num_classes):            # 跳过背景类
                cls_probs = probs[:, c]
                cls_deltas = bbox_deltas[:, c * 4:(c + 1) * 4]
                decoded = decode_boxes(proposals, cls_deltas)
                decoded[:, 0::2].clamp_(0, image_size[1])
                decoded[:, 1::2].clamp_(0, image_size[0])

                mask = cls_probs > 0.3
                if mask.sum() == 0:
                    continue
                boxes_c = decoded[mask]
                scores_c = cls_probs[mask]
                keep = nms(boxes_c, scores_c, 0.3)
                all_boxes.append(boxes_c[keep])
                all_labels.append(torch.full((len(keep),), c, device=feat.device))
                all_scores.append(scores_c[keep])

            if all_boxes:
                results.append({
                    'boxes': torch.cat(all_boxes),
                    'labels': torch.cat(all_labels),
                    'scores': torch.cat(all_scores),
                })
            else:
                results.append({'boxes': torch.empty(0, 4, device=feat.device),
                                'labels': torch.empty(0, device=feat.device),
                                'scores': torch.empty(0, device=feat.device)})
        return results


# ============================================================
#  7. 合成数据集 (用于 Demo)
# ============================================================
class SyntheticDataset(torch.utils.data.Dataset):
    """
    生成包含彩色矩形的合成图片 + ground truth 标注。
    class 1 = 红色方块,  class 2 = 蓝色方块
    用于快速验证 Faster R-CNN 流水线是否正确。
    """

    def __init__(self, num_samples=200, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        S = self.img_size
        img = torch.rand(3, S, S) * 0.3 + 0.1        # 灰色噪声背景

        boxes, labels = [], []
        n_objects = torch.randint(1, 4, (1,)).item()   # 1~3 个物体
        for _ in range(n_objects):
            cls = torch.randint(1, 3, (1,)).item()     # 1 或 2
            w = torch.randint(30, 70, (1,)).item()
            h = torch.randint(30, 70, (1,)).item()
            x1 = torch.randint(0, S - w, (1,)).item()
            y1 = torch.randint(0, S - h, (1,)).item()
            x2, y2 = x1 + w, y1 + h

            if cls == 1:     # 红色方块
                img[0, y1:y2, x1:x2] = 0.9
                img[1, y1:y2, x1:x2] = 0.1
                img[2, y1:y2, x1:x2] = 0.1
            else:            # 蓝色方块
                img[0, y1:y2, x1:x2] = 0.1
                img[1, y1:y2, x1:x2] = 0.1
                img[2, y1:y2, x1:x2] = 0.9

            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
        }
        return img, target


# ============================================================
#  8. 训练 & 推理 Demo
# ============================================================
def train_demo():
    """在合成数据上训练 Mini Faster R-CNN，验证流水线正确性。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # --- 参数 ---
    NUM_CLASSES = 3   # 背景 + 红色 + 蓝色
    NUM_EPOCHS = 15
    LR = 1e-3
    BATCH_SIZE = 4

    # --- 数据 ---
    dataset = SyntheticDataset(num_samples=200, img_size=256)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    # --- 模型 ---
    model = MiniFasterRCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- 训练循环 ---
    print("\n" + "=" * 60)
    print("  Mini Faster R-CNN — 训练开始")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for batch_idx, (images, targets) in enumerate(loader):
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            total_loss = sum(losses.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg = epoch_loss / len(loader)
        detail = " | ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
        print(f"  Epoch [{epoch:02d}/{NUM_EPOCHS}]  Loss: {avg:.4f}  ({detail})")

    # --- 推理 Demo ---
    print("\n" + "=" * 60)
    print("  推理测试")
    print("=" * 60)

    model.eval()
    # 取几张测试图
    test_dataset = SyntheticDataset(num_samples=5, img_size=256)
    for i in range(len(test_dataset)):
        img, gt = test_dataset[i]
        img_batch = img.unsqueeze(0).to(device)
        results = model(img_batch)
        det = results[0]
        n_det = det['boxes'].shape[0]
        print(f"\n  图片 {i+1}:")
        print(f"    GT boxes:  {gt['boxes'].shape[0]} 个  labels={gt['labels'].tolist()}")
        print(f"    检测到:    {n_det} 个")
        if n_det > 0:
            for j in range(min(n_det, 5)):
                box = det['boxes'][j].cpu().tolist()
                lbl = det['labels'][j].cpu().item()
                scr = det['scores'][j].cpu().item()
                cls_name = {1: '红色方块', 2: '蓝色方块'}.get(lbl, '未知')
                print(f"      [{cls_name}] score={scr:.3f}  box=["
                      f"{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

    print("\n" + "=" * 60)
    print("  ✅ Mini Faster R-CNN Demo 完成!")
    print("=" * 60)
    return model


# ============================================================
if __name__ == '__main__':
    train_demo()
