# Object Detection in an Urban Environment

### Project overview
**_Note: This is a submission for the Object Detection in an Urban Environment project, which is part of the [Udacity Self Driving Car Engineer Nano Degree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd0013)._**

Object detection in urban environments is critical for self-driving cars, because it allows the car to understand its surrounding and plan accordingly. For example, the car may slow down after detecting a pedestrian at a crossing or apply emergency breaks if it detects that it is on a collision course with another vehicle.

This project uses Tensorflow to train a machine learning model on samples from the [Waymo Open Dataset](https://waymo.com/open/). The projects covers:

* Analysing the dataset.
* Determining the data split for training, validation and testing.
* Modifying the provided model to improve the results.

---

### Set up
#### Project Structure
```
build/
    - Dockerfile
    - README.md
    - requirements.txt
experiments/
    - exporter_main_v2.py - to create an inference model
    - label_map.pbtxt - provides numerical ids for the classes in the dataset
    - model_main_tf2.py - to launch training
    - training/
        - pretrained-models/
        - reference/ - reference training with the unchanged config file
        - experiment0/  - a new folder should be created for each new experiment
        - experiment1/
        - experiment2/
        - ...
    ...
create_splits.py - used to create the data splits for training, validation and testing
download_process.py - used to download the dataset
edit_config.py - used to modify the pretrained model configuration file
inference_video.py - used to create a video of the model in action
utils.py - various convenience functions
Exploratory Data Analysis.ipynb - an analysis of the dataset
Explore augmentations.ipynb - used to visualise the augmentations in a given configuration file
filenames.txt - a list of all the dataset files' names
dark_files.txt - a list of the files containing dark scenes
rainy_files.txt - a list of the files containing rainy scenes
...
```

#### Environment Setup
Follow the instructions in `build/README.md` to set up the docker container with the required packages. **The following instructions should be executed from within the container.**

#### Downloading and processing the data
Run the following command to download and process the data:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

#### Creating the splits
To create the splits, run the following command:
```
python create_splits.py --data-dir data/
```
Note that the splits are based on the files in `dark_files.txt` and `rainy_files.txt`. The rationale behind the splits is covered later in this document.

#### Downloading and configuring a pretrained model
The project assumes that a pretrained model will be used as the starting point. To use the SSD Resnet 50 640x640 model:

1. Download the model from this [link](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz).
2. Extract the model under `pretrained-models/`, such that it matches the following hierarchy:
```
...
experiments/
    ...
    - training/
        ...
        - pretrained-models/
            - ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
        ...
    ...
...
```
3. Run the following command to create a configuration file with the split data locations and a modified batch size:
```
python edit_config.py --train_dir data/train/ --eval_dir data/val/ --batch_size 4 --checkpoint experiments/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map experiments/label_map.pbtxt
```
A new configuration file will be created.

_Note: for a list of other pretrained models, check this [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)._

#### Training & Validation
This section assumes the following file hierarchy:
```
...
experiments/
    ...
    - training/
        ...
        - experimentx/
            - pipeline.config
        ...
    ...
...
```
To train the model, run the following command:
```
python experiments/model_main_tf2.py --model_dir=experiments/experimentx/ --pipeline_config_path=experiments/experimentx/pipeline.config
```

To evaluate model, run the following command **in parallel with the training**:
```
python experiments/model_main_tf2.py --model_dir=experiments/experimentx/ --pipeline_config_path=experiments/experimentx/pipeline.config --checkpoint_dir=experiments/experimentx/
```

**_Note: if you do not have sufficient GPU resources, you may  not be able to run the training and validation scripts at the same time, one way to remedy this is to run the validation script on the CPU by using the following command instead:_**
<pre>
<b>CUDA_VISIBLE_DEVICES="-1"</b> python experiments/model_main_tf2.py --model_dir=experiments/experimentx/ --pipeline_config_path=experiments/experimentx/pipeline.config --checkpoint_dir=experiments/experimentx/
</pre>

Use tensorboard to visualise the training/validation progress:
```
tensorboard --logdir=experiments/training
```

---

### Dataset
#### Dataset analysis
The dataset contains 1997 images in various day and weather conditions:
<p float="left" align="middle">
  <img src="images/sample_1.png" width="300" />
  <img src="images/sample_2.png" width="300" /> 
  <img src="images/sample_3.png" width="300" />
</p>

The images are annotated with 3 classes: pedestrian, vehicle and cyclist. However, these classes are not equally represented in the dataset:
<p float="left" align="middle">
  <img src="images/class_dist.png" />
</p>

As shown in the graph, the number of vehicles severely outweighs that of the pedestrians and the cyclists. There is a very low number of cyclists, relative to the other classes. This inequality is likely to negatively impact the training and lead to poor generalisation.

The mean pixel value of each of the images was also investigated. The following figure shows that the mean pixel values are normally distributed, rather than being uniformly distributed. This may be an indicator that the majority of the photos have average lighting and only a few photos are on the brighter and the darker ends of the spectrum. As a results, the model may generalise poorly in different lighting conditions.

<p float="left" align="middle">
  <img src="images/pixel_dist.png" />
</p>

Finally, 1 image from each of the 100 files was visualised to get a sense of the variability in the environment. The images are not included here, but they can be viewed in `Exploratory Data Analysis.ipnyb`. There are some noteworthy observations from this visualisation:
1. Only 8 files contained dark scenes.
2. Only 11 files contained rainy scenes.
3. The rest of the 81 files contained scenes with mostly clear skies and daylight.


---
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
