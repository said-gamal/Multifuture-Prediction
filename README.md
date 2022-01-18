
# Multi-future Trajectory Prediction


This is the graduation project of ITI(AI-Pro) diploma.

The project is about using the [Multiverse](https://github.com/JunweiLiang/Multiverse) model 
and making it ready for deploying
in a slow-driving car by adding the required preprocessing like :

- annotation and object tracking (by using YOLOv5)
- semantic segmentation (by using DeepSORT) 
and making the whole preprocess automated.






## Pipeline
the pipeline in local machine for testing and evaluation :

![pipeline](readme_res\pipeline.PNG)


## Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1djvYdiGALytXtPYzYEGwSLzoRH7QA3AJ?usp=sharing)

Inferencing on a real video.

## Model output
final model output on a test video :

<video src='readme_res\download.mp4' width=900/>