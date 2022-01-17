# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# limit the number of cpus used by high performance libraries
import sys
sys.path.insert(0, './yolov5')

import numpy as np
from yolov5.utils.augmentations import letterbox
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import cv2
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("video_path")
parser.add_argument("--track_id", type=int)


def get_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_path = os.path.join('img_lst', video_name)
    if not os.path.exists(target_path):
      os.makedirs(target_path)
    cur_frame = 0
    img_lst_str = ''
    while(True):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            img_path = os.path.join(target_path, "%s_F_%08d.jpg" % (video_name, cur_frame))
            cv2.imwrite(img_path, frame)
            img_lst_str += str(img_path) + '\n'
            cur_frame += 1
        else:
            print('Captured frames:', len(frames))
            with open(f'{video_name}.lst', 'w') as out_file:
                out_file.write(img_lst_str)
            cap.release()
            return frames

def get_video_annotations(captured_frames, agent_track_id):
    annotation=[]
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
        pred = non_max_suppression(pred, 0.4, 0.5, classes=[0, 2, 3, 5, 6, 7], agnostic=False, max_det=1000)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s.copy()
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
                    x1 = output[0]
                    y1 = output[1]
                    x2 = output[2] - x1
                    y2 = output[3] - y1
                    bbox = [x1, y1, x2, y2]
                    if names[class_id] == 'person' and track_id==agent_track_id:
                        is_agent = 1
                    else:
                        is_agent = 0
                    output_text = {"class_name": str(names[class_id]).capitalize(), "is_x_agent": is_agent, "bbox": bbox,"frame_id": frame_id-2, "track_id": track_id}
                    annotation.append(output_text)             
            else:
                deepsort.increment_ages()
    return annotation

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
    video_frames = get_video_frames(args.video_path)
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    annotation = get_video_annotations(video_frames, agent_track_id=args.track_id)
    with open(f"{video_name}_annotations.json", "w") as outfile:
        outfile.write(str(annotation).replace('\'', '\"'))