import argparse
import os
import logging
from pathlib import Path
from threading import Thread
import yaml
from tqdm import tqdm

import numpy as np
import torch

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, set_logging, colorstr
from utils.torch_utils import select_device


logger = logging.getLogger(__name__)

def get_thres(data,
              cfg=None,
              weight=None,
              batch_size=32,
              imgsz=640,
              augment=False,
              half_precision=True):
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Load model
    model = Model(cfg, ch=3, nc=nc)  # create
    state_dict = torch.load(weight, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=True)  # load
    model.to(device)
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weight))  # report
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    model.get_score = True

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    model.eval()

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]

    score_list = []
    for batch_i, (img, _, _, _) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        with torch.no_grad():
            # Run model
            cur_score = model(img, augment=augment)  # inference and training outputs
            score_list.append(cur_score.item())

    thres = ['0']
    for i in list(range(10, 100, 10)):
        thres.append(str(np.percentile(score_list, i)))
    thres.append('1')
    return thres


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--weight', type=str, default='', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.single_cls = False
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ('train', 'val', 'test'):  # run normally
        thres = get_thres(opt.data, opt.cfg, opt.weight, opt.batch_size, opt.img_size, opt.augment)
        print()
        print('***************************************************')
        print(' '.join(thres))
        for idx, thr in enumerate(thres):
            print('First: {}%\tSecond: {}%\tThreshold: {}'.format(100 - idx * 10, idx  * 10, thr))
        print('***************************************************')
    else:
        raise NotImplementedError
