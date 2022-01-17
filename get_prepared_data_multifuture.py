# coding=utf-8
"""Given the final dataset or the anchor dataset, compile prepared data."""

import argparse
import json
import os
import operator
import pickle
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--start_frame", type=int)
parser.add_argument("--end_frame", type=int)
parser.add_argument("video_name")
parser.add_argument("annotations_path")
parser.add_argument("outpath_obs")
parser.add_argument("outpath_multifuture")


def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def convert_bbox(bbox):
  x, y, w, h = bbox
  return [x, y, x + w, y + h]

def get_feet(bbox):
  x1, y1, x2, y2 = bbox
  return ((x1 + x2) / 2.0, y2)

def get_frame_data(bbox_json):
  with open(bbox_json, "r") as f:
    bboxes = json.load(f)
  bboxes = filter_neg_boxes(bboxes)
  frame_data = {}  # frame_idx -> data
  for one in bboxes:
    if one["frame_id"] not in frame_data:
      frame_data[one["frame_id"]] = []
    frame_data[one["frame_id"]].append(one)

  return frame_data

def filter_neg_boxes(bboxes):
  new_bboxes = []
  for bbox in bboxes:
    x, y, w, h = bbox["bbox"]
    coords = x, y, x + w, y + h
    bad = False
    for o in coords:
      if o < 0:
        bad = True
    if not bad:
      new_bboxes.append(bbox)
  return new_bboxes

if __name__ == "__main__":
  args = parser.parse_args()

  args.obs_length = int((args.end_frame - args.start_frame) / 10) + 1
  args.drop_frame = {"virat": 10, "ethucy": 10}
  # we want the obs to be 3.2 sec long
  args.frame_range = {"virat": (args.start_frame, args.end_frame)}
  class2classid = {"Person": 0, "Vehicle": 1}

  args.traj_path = os.path.join(args.outpath_obs, "traj_2.5fps")
  args.person_box_path = os.path.join(args.outpath_obs, "anno_person_box")
  args.other_box_path = os.path.join(args.outpath_obs, "anno_other_box")
  
  # for split in tqdm(filelst, ascii=True):
  for split in tqdm([args.video_name], ascii=True):
    traj_path = os.path.join(args.traj_path, split)
    mkdir(traj_path)
    person_box_path = os.path.join(args.person_box_path, split)
    mkdir(person_box_path)
    other_box_path = os.path.join(args.other_box_path, split)
    mkdir(other_box_path)
    multifuture_path = os.path.join(args.outpath_multifuture, split)
    mkdir(multifuture_path)
    # cluster the obs
    # obs_id -> filenames
    # unique_obs_videonames = get_obs_videonames(filelst[split])
    unique_obs_videonames = [f'{args.video_name}_{args.start_frame}_{args.end_frame-1}']
    if not unique_obs_videonames:
      print("skipping empty split %s" % split)
      continue
    skipped_count = 0
    future_timestep_counts = []
    for obs_key in tqdm(unique_obs_videonames):
      videoname = unique_obs_videonames[0]
      drop_frame = args.drop_frame["virat"]
      start_frame, end_frame = args.frame_range["virat"]
      bbox_json = args.annotations_path
      frame_data = get_frame_data(bbox_json)
      frame_idxs = sorted(frame_data.keys())
      obs_frame_idxs = frame_idxs[start_frame:end_frame:drop_frame]
      # 2. gather data for each frame_idx, each person_idx
      traj_data = []  # [frame_idx, person_idx, x, y]
      person_box_data = {}  # (frame_idx, person_id) -> boxes
      other_box_data = {}  # (frame_idx, person_id) -> other boxes + boxclasids
      obs_x_agent_traj = []
      for frame_idx in obs_frame_idxs:
        box_list = frame_data[frame_idx]
        box_list.sort(key=operator.itemgetter("track_id"))
        for i, box in enumerate(box_list):
          class_name = box["class_name"]
          track_id = box["track_id"]
          is_x_agent = box["is_x_agent"]
          bbox = convert_bbox(box["bbox"])
          if class_name == "Person":
            new_frame_idx = frame_idx - start_frame
            person_key = "%d_%d" % (new_frame_idx, track_id)
            x, y = get_feet(bbox)
            traj_data.append((new_frame_idx, float(track_id), x, y))
            if int(is_x_agent) == 1:
              obs_x_agent_traj.append((new_frame_idx, float(track_id), x, y))
            person_box_data[person_key] = bbox
            all_other_boxes = [convert_bbox(box_list[j]["bbox"]) for j in range(len(box_list)) if j != i]
            all_other_boxclassids = [class2classid[box_list[j]["class_name"]] for j in range(len(box_list)) if j != i]
            other_box_data[person_key] = (all_other_boxes, all_other_boxclassids)
      if len(obs_x_agent_traj) != args.obs_length:
        # print(obs_x_agent_traj)
        print("warning, skipping %s due to bad x_agent boxes" % videoname)
        skipped_count += 1
        continue
      # save the data
      desfile = os.path.join(traj_path, "%s.txt" % (videoname))
      delim = "\t"
      # print(traj_data)
      with open(desfile, "w") as f:
        for i, p, x, y in traj_data:
          f.writelines("%d%s%.1f%s%.6f%s%.6f\n" % (i, delim, p, delim, x, delim, y))
      with open(os.path.join(person_box_path, "%s.p" % videoname), "wb") as f:
        pickle.dump(person_box_data, f)
      with open(os.path.join(other_box_path, "%s.p" % videoname), "wb") as f:
        pickle.dump(other_box_data, f)
      # now we save all the multi future paths for all agent.
      multifuture_data = {}  # videoname -> {"x_agent_traj", "all_boxes"}
      for videoname in unique_obs_videonames:
        # bbox_json = os.path.join(args.dataset_path, "bbox", "%s.json" % videoname)
        frame_data = get_frame_data(bbox_json)
        frame_idxs = sorted(frame_data.keys())
        assert frame_idxs[0] == 0
        # 1. first pass, get the needed frames
        needed_frame_idxs = frame_idxs[start_frame::drop_frame]
        assert len(needed_frame_idxs) > args.obs_length, (needed_frame_idxs, start_frame)
        pred_frame_idxs = needed_frame_idxs[args.obs_length:]
        future_timestep_counts.append(len(pred_frame_idxs))
        x_agent_traj = []
        all_boxes = []
        for frame_idx in pred_frame_idxs:
          box_list = frame_data[frame_idx]
          box_list.sort(key=operator.itemgetter("track_id"))
          for i, box in enumerate(box_list):
            class_name = box["class_name"]
            track_id = box["track_id"]
            is_x_agent = box["is_x_agent"]
            bbox = convert_bbox(box["bbox"])
            new_frame_idx = frame_idx - start_frame
            if int(is_x_agent) == 1:
              x, y = get_feet(bbox)
              x_agent_traj.append((new_frame_idx, track_id, x, y))
            all_boxes.append((new_frame_idx, class_name, is_x_agent, track_id, bbox))
        multifuture_data[videoname] = {
            "x_agent_traj": x_agent_traj, # future
            "all_boxes": all_boxes,
            "obs_traj": obs_x_agent_traj,
        }
      target_file = os.path.join(multifuture_path, "%s.p" % videoname)
      with open(target_file, "wb") as f:
        pickle.dump(multifuture_data, f)
    print("original obs %s, skipped %s" % (len(unique_obs_videonames), skipped_count))