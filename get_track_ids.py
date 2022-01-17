# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
sys.path.insert(0, './yolov5')

import numpy as np
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import cv2
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video_path")


def get_first_two_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    cur_frame = 0
    while(cur_frame < 20):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            cur_frame += 1
    cap.release()
    return frames

def save_track_ids(captured_frames):
    yolo_weights= 'yolov5l.pt'
    imgsz= [640]
    imgsz *= 2 if len(imgsz) == 1 else 1
    half = False
    dnn=False
    device=''
    # initialize deepsort
    deepsort = initialize_deepsort()
    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn)
    stride, names, pt, jit = model.stride, model.names, model.pt, model.jit
    # set super class name vehicle
    for i in [2, 3, 5, 6, 7]:
        names[i] = 'vehicle'
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # extract what is in between the last '/' and last '.'
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    # get annotations by frame
    for frame_id, frame in enumerate(captured_frames):
        # Padded resize
        frame_img = letterbox(frame, 640, 32, True)[0]
        # Convert
        frame_img = frame_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame_img = np.ascontiguousarray(frame_img)
        img, im0s = frame_img, frame
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = model(img, augment=False, visualize=False)
        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.5, classes=[0, 2, 3, 5, 6, 7], agnostic=False, max_det=1000)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                # get annotations
                for output in outputs: 
                    track_id = output[4]
                    class_id = int(output[5])  # integer class
                    annotator.box_label(output[:4], f'id:{track_id}', color=colors(class_id, True))            
            else:
                deepsort.increment_ages()
            im0 = annotator.result()
            cv2.imwrite('track_ids.jpg', im0)

def initialize_deepsort():
    cfg = get_config()
    cfg.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)             
    return deepsort

if __name__ == "__main__":
    args = parser.parse_args()
    video_frames = get_first_two_frames(args.video_path)
    save_track_ids(video_frames)