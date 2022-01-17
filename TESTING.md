## Multi-Future Trajectory Prediction
In this section we show how to do multi-future trajectory inferencing on a real video.

First download the pretrained models `Deeplab` and `Multiverse`.
```
$ wget http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz
$ tar -zxvf deeplabv3_xception_ade20k_train_2018_05_29.tar.gz; rm deeplabv3_xception_ade20k_train_2018_05_29.tar.gz
$ wget https://next.cs.cmu.edu/multiverse/dataset/multiverse-models.tgz
$ tar -zxvf multiverse-models.tgz; rm multiverse-models.tgz
```

Suppose you have the video in the root folder named `test_video.mp4`.

1- Run the following script to get `track_ids.jpg` to select the person you track:
```
$ python get_track_ids.py test_video.mp4
```

This will save the `track_ids.jpg` image in the repo folder.

2- Run the following script to do annotation by `Yolov5`:
```
$ python annotation_by_yolo.py test_video.mp4 --track_id 10
```

This will save the `test_video_annotations.json` file in the root folder.

3- Run the following script to do semantic segmentation by pretrained `Deeplab` model:
```
$ python semantic_segmentation.py test_video.lst deeplabv3_xception_ade20k_train/frozen_inference_graph.pb \
seg_36x64 --every 10 --down_rate 8.0
```

This will create `seg_36x64` folder in the root folder contains the `npy` files.

4- Run the following script to prepare the data for the model:
```
$ python prepare_data.py 600 test_video test_video_annotations.json prepared_data/obs_data prepared_data/multifuture
```

This will create `prepared_data` folder in the root folder contains the `multifuture` and `obs_data` files.

5- Now run the following script to run the `model` on the prepared data:
```
$ python multifuture_inference.py prepared_data/obs_data/traj_2.5fps/test_video/ prepared_data/multifuture/test_video/ \
multiverse-models/multiverse_single18.51_multi168.9_nll2.6/00/best/ test_video_output.traj.p --save_prob_file test_video_output.prob.p \
--obs_len 10 --emb_size 32 --enc_hidden_size 256 --dec_hidden_size 256 --use_scene_enc --scene_id2name \
scene36_64_id2name_top10.json --scene_feat_path seg_36x64/test_video --scene_h 36 --scene_w 64 \
--scene_conv_kernel 3 --scene_conv_dim 64 --grid_strides 2,4 --use_grids 1,0 --num_out 20 \
--diverse_beam --diverse_gamma 0.01 --fix_num_timestep 1 --gpuid 0
```

This will save the model output files `test_video_output.traj.p` and `test_video_output.prob.p` in the root folder.

## Visualization
To visualize the model output, run:
```
$ python visualize_output.py prepared_data/multifuture/test_video/ test_video_output.traj.p ./ \
model_output_visualize_videos/test_video --show_obs --use_heatmap --drop_frame 10
$ cd model_output_visualize_videos
$ for file in *;do ffmpeg -framerate 4 -i ${file}/%08d.jpg ${file}.mp4;done
```
