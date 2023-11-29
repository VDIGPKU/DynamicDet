import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, box_iou, xywh2xyxy
from utils.torch_utils import is_parallel


def smooth_BCE(
    eps=0.1
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t)**self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super().__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get(
            'label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model_h2[-1] if is_parallel(
            model) else model.model_h2[-1]  # Detect() module
        self.balance = {
            3: [4.0, 1.0, 0.4]
        }.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(
            det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p,
                                                          targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj,
                        gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False,
                               CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj,
                     gi] = (1.0 -
                            self.gr) + self.gr * iou.detach().clamp(0).type(
                                tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn,
                                        device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[
                    i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(
            7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na,
                          device=targets.device).float().view(na, 1).repeat(
                              1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]),
                            2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1),
                 gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLossOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super().__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get(
            'label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model_h2[-1] if is_parallel(
            model) else model.model_h2[-1]  # Detect() module
        self.balance = {
            3: [4.0, 1.0, 0.4]
        }.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(
            det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            p, targets, imgs)
        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p
        ]

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[
                i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj,
                        gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T,
                               selected_tbox,
                               x1y1x2y2=False,
                               CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj,
                     gi] = (1.0 -
                            self.gr) + self.gr * iou.detach().clamp(0).type(
                                tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn,
                                        device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[
                    i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):

        # indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)
        # indices, anch = self.find_4_positive(p, targets)
        # indices, anch = self.find_5_positive(p, targets)
        # indices, anch = self.find_9_positive(p, targets)

        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)

        for batch_idx in range(p[0].shape[0]):

            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(p):

                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b), )) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 +
                       grid) * self.stride[i]  # / 8.
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() *
                       2)**2 * anch[i][idx] * self.stride[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou,
                                  min(10, pair_wise_iou.shape[1]),
                                  dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (F.one_hot(this_target[:, 1].to(torch.int64),
                                          self.nc).float().unsqueeze(1).repeat(
                                              1, pxyxys.shape[0], 1))

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() *
                p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image,
                reduction='none').sum(-1)
            del cls_preds_

            cost = (pair_wise_cls_loss + 3.0 * pair_wise_iou_loss)

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx],
                                        k=dynamic_ks[gt_idx].item(),
                                        largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1],
                                           dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([],
                                              device='cuda:0',
                                              dtype=torch.int64)
                matching_as[i] = torch.tensor([],
                                              device='cuda:0',
                                              dtype=torch.int64)
                matching_gjs[i] = torch.tensor([],
                                               device='cuda:0',
                                               dtype=torch.int64)
                matching_gis[i] = torch.tensor([],
                                               device='cuda:0',
                                               dtype=torch.int64)
                matching_targets[i] = torch.tensor([],
                                                   device='cuda:0',
                                                   dtype=torch.int64)
                matching_anchs[i] = torch.tensor([],
                                                 device='cuda:0',
                                                 dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(
            7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na,
                          device=targets.device).float().view(na, 1).repeat(
                              1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]),
                            2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            # gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1),
                 gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLossOTADual(ComputeLossOTA):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super().__init__(model, autobalance)

    def __call__(self, p, targets, imgs):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device)

        bs_2, as_2_, gjs_2, gis_2, targets_2, anchors_2 = self.build_targets(
            p[self.nl:], targets, imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            p[:self.nl], targets, imgs)
        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]]
            for pp in p[:self.nl]
        ]
        pre_gen_gains_2 = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]]
            for pp in p[self.nl:]
        ]

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_2 = p[i + self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[
                i]  # image, anchor, gridy, gridx
            b_2, a_2, gj_2, gi_2 = bs_2[i], as_2_[i], gjs_2[i], gis_2[
                i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_2 = torch.zeros_like(pi_2[..., 0],
                                      device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj,
                        gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T,
                               selected_tbox,
                               x1y1x2y2=False,
                               CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj,
                     gi] = (1.0 -
                            self.gr) + self.gr * iou.detach().clamp(0).type(
                                tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn,
                                        device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            n_2 = b_2.shape[0]  # number of targets
            if n_2:
                ps_2 = pi_2[b_2, a_2, gj_2,
                            gi_2]  # prediction subset corresponding to targets

                # Regression
                grid_2 = torch.stack([gi_2, gj_2], dim=1)
                pxy_2 = ps_2[:, :2].sigmoid() * 2. - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh_2 = (ps_2[:, 2:4].sigmoid() * 2)**2 * anchors_2[i]
                pbox_2 = torch.cat((pxy_2, pwh_2), 1)  # predicted box
                selected_tbox_2 = targets_2[i][:, 2:6] * pre_gen_gains_2[i]
                selected_tbox_2[:, :2] -= grid_2
                iou_2 = bbox_iou(pbox_2.T,
                                 selected_tbox_2,
                                 x1y1x2y2=False,
                                 CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou_2).mean()  # iou loss

                # Objectness
                tobj_2[b_2, a_2, gj_2, gi_2] = (
                    1.0 - self.gr) + self.gr * iou_2.detach().clamp(0).type(
                        tobj_2.dtype)  # iou ratio

                # Classification
                selected_tcls_2 = targets_2[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_2 = torch.full_like(ps_2[:, 5:], self.cn,
                                          device=device)  # targets
                    t_2[range(n_2), selected_tcls_2] = self.cp
                    lcls += self.BCEcls(ps_2[:, 5:], t_2)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_2 = self.BCEobj(pi_2[..., 4], tobj_2)
            lobj += obji * self.balance[i]  # obj loss
            lobj += obji_2 * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[
                    i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


class ComputeLossOTADy(ComputeLossOTA):
    # Compute losses
    def __init__(self, model, autobalance=False):
        super().__init__(model, autobalance)
        self.tracked_diff = 0
        self.iter_count = 0
        self.diff_list = []

    def __call__(self, ps, targets, imgs):  # predictions, targets, model
        p, score = ps
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device)
        lcls_2, lbox_2, lobj_2 = torch.zeros(1, device=device), torch.zeros(
            1, device=device), torch.zeros(1, device=device)

        bs_2, as_2_, gjs_2, gis_2, targets_2, anchors_2 = self.build_targets(
            p[self.nl:], targets, imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(
            p[:self.nl], targets, imgs)
        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]]
            for pp in p[:self.nl]
        ]
        pre_gen_gains_2 = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]]
            for pp in p[self.nl:]
        ]

        # Losses
        for i in range(self.nl):  # layer index, layer predictions
            pi = p[i]
            pi_2 = p[i + self.nl]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[
                i]  # image, anchor, gridy, gridx
            b_2, a_2, gj_2, gi_2 = bs_2[i], as_2_[i], gjs_2[i], gis_2[
                i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_2 = torch.zeros_like(pi_2[..., 0],
                                      device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj,
                        gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2)**2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T,
                               selected_tbox,
                               x1y1x2y2=False,
                               CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj,
                     gi] = (1.0 -
                            self.gr) + self.gr * iou.detach().clamp(0).type(
                                tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn,
                                        device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            n_2 = b_2.shape[0]  # number of targets
            if n_2:
                ps_2 = pi_2[b_2, a_2, gj_2,
                            gi_2]  # prediction subset corresponding to targets

                # Regression
                grid_2 = torch.stack([gi_2, gj_2], dim=1)
                pxy_2 = ps_2[:, :2].sigmoid() * 2. - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh_2 = (ps_2[:, 2:4].sigmoid() * 2)**2 * anchors_2[i]
                pbox_2 = torch.cat((pxy_2, pwh_2), 1)  # predicted box
                selected_tbox_2 = targets_2[i][:, 2:6] * pre_gen_gains_2[i]
                selected_tbox_2[:, :2] -= grid_2
                iou_2 = bbox_iou(pbox_2.T,
                                 selected_tbox_2,
                                 x1y1x2y2=False,
                                 CIoU=True)  # iou(prediction, target)
                lbox_2 += (1.0 - iou_2).mean()  # iou loss

                # Objectness
                tobj_2[b_2, a_2, gj_2, gi_2] = (
                    1.0 - self.gr) + self.gr * iou_2.detach().clamp(0).type(
                        tobj_2.dtype)  # iou ratio

                # Classification
                selected_tcls_2 = targets_2[i][:, 1].long()
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t_2 = torch.full_like(ps_2[:, 5:], self.cn,
                                          device=device)  # targets
                    t_2[range(n_2), selected_tcls_2] = self.cp
                    lcls_2 += self.BCEcls(ps_2[:, 5:], t_2)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_2 = self.BCEobj(pi_2[..., 4], tobj_2)
            lobj += obji * self.balance[i]  # obj loss
            lobj_2 += obji_2 * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[
                    i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lbox_2 *= self.hyp['box']
        lobj_2 *= self.hyp['obj']
        lcls_2 *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = (lbox + lobj + lcls).detach()
        loss_2 = (lbox_2 + lobj_2 + lcls_2).detach()

        self.iter_count += 1
        current_diff = loss - loss_2
        self.diff_list.append(current_diff.item())

        # (online) calculation of the loss difference
        if self.iter_count < 10000:
            self.tracked_diff = np.median(self.diff_list)
        elif self.iter_count % 1000 == 0:
            self.tracked_diff = np.median(self.diff_list)

        loss = loss - self.tracked_diff / 2
        loss_2 = loss_2 + self.tracked_diff / 2
        adaptive_loss = score[:, 0] * loss + (1 - score[:, 0]) * loss_2
        return adaptive_loss * bs, torch.cat(
            (current_diff, score[:,
                                 0], 1 - score[:, 0], adaptive_loss)).detach()
