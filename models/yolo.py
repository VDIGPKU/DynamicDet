import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.nn.modules.batchnorm import _BatchNorm

from models.common import (NMS, SPPCSPC, AdaptiveRouter, CBFuse, CBLinear,
                           Concat, Conv, ConvCheckpoint, ImplicitA, ImplicitM,
                           ReOrg, RepConv, Shortcut, autoShape)
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights,
                               model_info, scale_img, time_synchronized)

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid',
                             a.clone().view(self.nl, 1, -1, 1, 1,
                                            2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                               for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny,
                             nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                               self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2)**2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny,
                             nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                                   self.grid[i]) * self.stride[i]  # xy
                    y[...,
                      2:4] = (y[..., 2:4] * 2)**2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split(
                        (2, 2, self.nc + 1),
                        4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (
                        self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh**2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    def fuse(self):
        print('IDetect.fuse')
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(
                self.m[i].weight.reshape(c1, c2),
                self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32,
            device=z.device)
        box @= convert_matrix
        return (box, score)


class Model(nn.Module):

    def __init__(self,
                 cfg,
                 ch=3,
                 nc=None):  # model, input channels, number of classes
        super().__init__()
        assert isinstance(cfg, str)
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        self.dynamic = self.yaml.get('dynamic', False)
        if self.dynamic:
            router_channels = self.yaml['router_channels']
            reduction = self.yaml.get('router_reduction', 4)
            self.router = AdaptiveRouter(router_channels,
                                         1,
                                         reduction=reduction)
            self.router_ins = self.yaml['router_ins']
            self.dy_thres = 0.5
            self.get_score = False

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(
                f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        (self.model_b, self.save_b, self.model_b2, self.save_b2, self.model_h,
         self.save_h, self.model_h2,
         self.save_h2) = parse_model(deepcopy(self.yaml),
                                     ch_b=[ch])  # model, savelist
        self.keep_input = self.yaml.get('keep_input', False)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model_h[-1]  # Detect()
        m2 = self.model_h2[-1]  # Detect()
        if isinstance(m, IDetect):
            s = 256  # 2x min stride
            if self.dynamic:
                m.stride = torch.tensor([
                    s / x.shape[-2] for x in self.forward(
                        torch.zeros(1, ch, s, s))[0][:m.anchors.shape[0]]
                ])  # forward
            else:
                m.stride = torch.tensor([
                    s / x.shape[-2] for x in self.forward(
                        torch.zeros(1, ch, s, s))[:m.anchors.shape[0]]
                ])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            if self.dynamic:
                m2.stride = torch.tensor([
                    s / x.shape[-2] for x in self.forward(
                        torch.zeros(1, ch, s, s))[0][:m2.anchors.shape[0]]
                ])  # forward
            else:
                m2.stride = torch.tensor([
                    s / x.shape[-2] for x in self.forward(
                        torch.zeros(1, ch, s, s))[:m2.anchors.shape[0]]
                ])  # forward
            check_anchor_order(m2)
            m2.anchors /= m2.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        # Init weights, biases
        initialize_weights(self)
        # self.initialize_cblinear()
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x,
                               si,
                               gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x,
                                     profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        if self.keep_input:
            input_x = x

        y, dt = [], []  # outputs
        outs = []
        for m in self.model_b:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(
                    m.f, int) else [x if j == -1 else y[j]
                                    for j in m.f]  # from earlier layers

            if profile:
                c = isinstance(m, IDetect)
                o = thop.profile(
                    m, inputs=(x.copy() if c else x, ),
                    verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run

            y.append(x if m.i in self.save_b else None)  # save output
        assert len(y) == self.yaml['n_first_layers']

        if self.dynamic:
            score = self.router([
                y[i] for i in self.router_ins
            ])  # 'score' denotes the (1 - difficulty score)

            if not hasattr(self, 'get_score'):
                self.get_score = False
            if self.get_score:
                return score

        need_second = self.training or (
            not self.dynamic) or score[:, 0] < self.dy_thres
        need_first_head = self.training or (self.dynamic
                                            and score[:, 0] >= self.dy_thres)

        if need_second:
            for m in self.model_b2:
                if m.f == 'input':
                    x = input_x
                elif m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(
                        m.f, int) else [x if j == -1 else y[j]
                                        for j in m.f]  # from earlier layers

                if profile:
                    c = isinstance(m, IDetect)
                    o = thop.profile(
                        m, inputs=(x.copy() if c else x, ),
                        verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' %
                          (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x if m.i in self.save_b2 else None)  # save output

        if need_first_head:
            for m in self.model_h:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(
                        m.f, int) else [x if j == -1 else y[j]
                                        for j in m.f]  # from earlier layers

                if profile:
                    c = isinstance(m, IDetect)
                    o = thop.profile(
                        m, inputs=(x.copy() if c else x, ),
                        verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' %
                          (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x if m.i in self.save_h else None)  # save output

            outs.extend(x)

        if need_second:
            for m in self.model_h2:
                if isinstance(m.f, int) and m.f > 0:
                    cur_f = m.f + self.yaml['n_first_layers']
                else:
                    cur_f = m.f
                if cur_f != -1:  # if not from previous layer
                    x = y[cur_f] if isinstance(cur_f, int) else [
                        x if j == -1 else y[j] for j in cur_f
                    ]  # from earlier layers

                if profile:
                    c = isinstance(m, IDetect)
                    o = thop.profile(
                        m, inputs=(x.copy() if c else x, ),
                        verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                    for _ in range(10):
                        m(x.copy() if c else x)
                    t = time_synchronized()
                    for _ in range(10):
                        m(x.copy() if c else x)
                    dt.append((time_synchronized() - t) * 100)
                    print('%10.1f%10.0f%10.1fms %-40s' %
                          (o, m.np, dt[-1], m.type))

                x = m(x)  # run

                y.append(x if m.i in self.save_h2 else None)  # save output

            outs.extend(x)

        if profile:
            print('%.1fms total' % sum(dt))
        if self.training and self.dynamic:
            return outs, score
        else:
            return outs

    def close_all_bn(self):
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def _initialize_biases(
            self,
            cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model_h[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s)**2)  # obj (8 objects per 640 image)
            b.data[:,
                   5:] += math.log(0.6 /
                                   (m.nc - 0.99)) if cf is None else torch.log(
                                       cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        m = self.model_h2[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s)**2)  # obj (8 objects per 640 image)
            b.data[:,
                   5:] += math.log(0.6 /
                                   (m.nc - 0.99)) if cf is None else torch.log(
                                       cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model_h[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) %
                  (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
        m = self.model_h2[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) %
                  (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for model in [
                self.model_b, self.model_b2, self.model_h, self.model_h2
        ]:
            for m in model.modules():
                if isinstance(m, RepConv):
                    # print(f" fuse_repvgg_block")
                    m.fuse_repvgg_block()
                elif type(m) is Conv and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.fuseforward  # update forward
                elif isinstance(m, IDetect):
                    m.fuse()
                    m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model_h[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model_h[-1].i + 1  # index
            self.model_h.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model_h = self.model_h[:-1]  # remove
        present = type(self.model_h2[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model_h2[-1].i + 1  # index
            self.model_h2.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model_h2 = self.model_h2[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m,
                  self,
                  include=('yaml', 'nc', 'hyp', 'names', 'stride'),
                  exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch_b):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' %
                ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d[
        'width_multiple']
    na = (len(anchors[0]) //
          2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers_b, save_b, c2 = [], [], ch_b[-1]  # layers, savelist, ch_b out

    for i, (f, n, m,
            args) in enumerate(d['backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except Exception:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            c1, c2 = ch_b[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch_b[f]]
        elif m is Concat:
            c2 = sum([ch_b[x] for x in f])
        elif m is Shortcut:
            c2 = ch_b[f[0]]
        elif m is IDetect:
            args.append([ch_b[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch_b[f] * 4
        else:
            c2 = ch_b[f]

        m_ = nn.Sequential(
            *[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n, np, t, args))  # print
        save_b.extend(x % i for x in ([f] if isinstance(f, (int, str)) else f)
                      if x != -1)  # append to savelist
        layers_b.append(m_)
        if i == 0:
            ch_b = []
        ch_b.append(c2)

    layers_b2, save_b2 = [], []  # layers, savelist
    ch_b2 = []

    for i, (f, n, m, args) in enumerate(
            d['dual_backbone']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except Exception:
                pass

        chs = []
        for x in ([f] if isinstance(f, (int, str)) else f):
            if isinstance(x, str):
                continue
            if x >= 0:
                chs.append(ch_b)
            else:
                chs.append(ch_b2)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            if f == 'input':
                c1, c2 = 3, args[0]
            else:
                assert len(chs) == 1
                c1, c2 = chs[0][f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            assert len(chs) == 1
            args = [chs[0][f]]
        elif m is Concat:
            c2 = sum([ch[x] for x, ch in zip(f, chs)])
        elif m is Shortcut:
            assert len(chs) == 1
            c2 = chs[0][f[0]]
        elif m is IDetect:
            args.append([ch[x] for x, ch in zip(f, chs)])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            assert len(chs) == 1
            c2 = chs[0][f] * 4
        elif m is CBLinear:
            c2 = args[0]
            assert len(chs) == 1
            c1 = chs[0][f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = chs[-1][f[-1]]
        else:
            assert len(chs) == 1
            c2 = chs[0][f]

        m_ = nn.Sequential(
            *[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n, np, t, args))  # print
        for x in ([f] if isinstance(f,
                                    (int, str)) else f):  # append to savelist
            if isinstance(x, str):
                continue
            if x >= 0:
                save_b.extend([x])
            elif x != -1:
                save_b2.extend([x % i])
        layers_b2.append(m_)
        ch_b2.append(c2)

    layers_h, save_h, ch_h = [], [], []
    for i, (f, n, m,
            args) in enumerate(d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except Exception:
                pass
        chs = []
        for x in ([f] if isinstance(f, (int, str)) else f):
            if isinstance(x, str):
                continue
            if x >= 0:
                chs.append(ch_b)
            else:
                chs.append(ch_h)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            assert len(chs) == 1
            c1, c2 = chs[0][f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            assert len(chs) == 1
            args = [chs[0][f]]
        elif m is Concat:
            c2 = sum([ch[x] for x, ch in zip(f, chs)])
        elif m is Shortcut:
            assert len(chs) == 1
            c2 = chs[0][f[0]]
        elif m is IDetect:
            args.append([ch[x] for x, ch in zip(f, chs)])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            assert len(chs) == 1
            c2 = chs[0][f] * 4
        else:
            assert len(chs) == 1
            c2 = chs[0][f]

        m_ = nn.Sequential(
            *[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n, np, t, args))  # print
        for x in ([f] if isinstance(f,
                                    (int, str)) else f):  # append to savelist
            if isinstance(x, str):
                continue
            if x >= 0:
                save_b.extend([x])
            elif x != -1:
                save_h.extend([x % i])
        layers_h.append(m_)
        ch_h.append(c2)

    layers_h2, save_h2, ch_h2 = [], [], []
    for i, (f, n, m,
            args) in enumerate(d['head2']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except Exception:
                pass
        chs = []
        for x in ([f] if isinstance(f, (int, str)) else f):
            if isinstance(x, str):
                continue
            if x >= 0:
                chs.append(ch_b2)
            else:
                chs.append(ch_h2)

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, ConvCheckpoint, RepConv, SPPCSPC]:
            assert len(chs) == 1
            c1, c2 = chs[0][f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            assert len(chs) == 1
            args = [chs[0][f]]
        elif m is Concat:
            c2 = sum([ch[x] for x, ch in zip(f, chs)])
        elif m is Shortcut:
            assert len(chs) == 1
            c2 = chs[0][f[0]]
        elif m is IDetect:
            args.append([ch[x] for x, ch in zip(f, chs)])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            assert len(chs) == 1
            c2 = chs[0][f] * 4
        else:
            assert len(chs) == 1
            c2 = chs[0][f]

        m_ = nn.Sequential(
            *[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n, np, t, args))  # print
        for x in ([f] if isinstance(f,
                                    (int, str)) else f):  # append to savelist
            if isinstance(x, str):
                continue
            if x >= 0:
                save_b.extend([x])
            elif x != -1:
                save_h2.extend([x % i])
        layers_h2.append(m_)
        ch_h2.append(c2)

    save_b.extend(d['b1_save'])
    save_b2.extend(d['b2_save'])

    return (nn.Sequential(*layers_b),
            sorted(save_b), nn.Sequential(*layers_b2), sorted(save_b2),
            nn.Sequential(*layers_h), sorted(save_h),
            nn.Sequential(*layers_h2), sorted(save_h2))
