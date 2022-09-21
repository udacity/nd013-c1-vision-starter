# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure
This project contains the necessary files for training an object detection resnet neural network. There are several python scripts that are useful for downloading training data and splitting it into train, validation and test datasets. Also there is a script to generate new config files to train the tensorflow model. There are two jupyter notebooks for visualization of the raw data and the augmented data respectively. 
### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

The following python libraries are required:
- Cython
- jupyter
- matplotlib
- Pillow
- ray
- waymo-open-dataset-tf-2-5-0

tensorflow and CUDA are required as well.
### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

## Submission Template

### Project overview
The aim of this project is to train an object detection model for a camera mounted on a car. Training data is taken out from the waymo open dataset. Object detection is a crucial feature of self driving cars because it allows to transform images into useful information providing the car with data about the relative position of objects that it has to avoid, preventing crashes and injuries. 

### Set up
For setting up the project you can follow the steps on the **Local Setup** section on this readme. Additionally you can use the [remote - containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) vscode extension to build the container and access it though vscode.

### Dataset
As stated above, this project uses the data from the waymo open dataset. Instructions to download it from google cloud storage are provided above as well.
#### Dataset analysis
The selected dataset includes a higly variable group of images from highways, residential areas, parking lots, daytime, nightime, rain, etc. The following pictures show 10 random images that were overlayed with the ground truth bounding boxes.

![img](exploratory_analysis.png)

Only cars, pedestrians and bicycles are included in the ground truth data. The most common objects are by far cars follwed by pedestrians with about a third of the occurrences. bicycles are very rare; just as the histogram below shows: 

![img](classes.png)

The dataset shows great color variability. RGB values are quite uniformely distrubuted with a slightly higher probability of being ~80 likely due to the very common apperance of the greyish color of the street in the images. most images have good brightness as the picture below shows:

![histogram](histogram.png)

#### Cross validation
The dataset was split in 3 datasets containing:

- 70% of the images for training
- 20% of the images for validation
- 10% of the images for testing

Since the dataset is relatively large it is possible to rely on a smaller proportion for cross validation.

### Training
#### Reference experiment
The first training session was conducted with the default presets for 25000 epochs (see [pipeline](experiments/reference/pipeline_new.config)). the following picture shows how the loss behaved:

![Screenshot from 2022-09-20 19-04-07](https://user-images.githubusercontent.com/71234974/191391728-0072ed40-d7d7-4b9b-9762-3f03ea82c9bd.png)

In the warmup phase the loss was fairly low and constant, however in the epoch 4k a huge increase occurred. After that the training continued effectively reducing the loss as more iterations were run, however the final results were very poor anyway due to this huge increase that happened over a short period of time. This behavior was probably due to a high learning rate, which took the model way out of its minimum in very few iterations (see picture below).

![image](https://user-images.githubusercontent.com/71234974/191393310-c00f8504-e865-468d-96ba-c366267aef9e.png)

As the blue dots show, validation losses are consistent with the training losses, which shows that even if the performance of the model is poor it did not overfit. These were the final metric: 

    - mAP@0.5IOU: 0.000063
    - AP@0.5IOU/vehicle: 0.000126
    - AP@0.5IOU/pedestrian: 0.000000
    - AP@0.5IOU/cyclist: nan
    - localization_loss: 0.889936
    - classification_loss: 130.260071
    - regularization_loss: 2002455621533696.000000
    - total_loss: 2002455621533696.000000

This metrics show once again that the performance of the model is poor. Cyclists AP is not reported likely due to the extremely low amount of cyclist in the datasets, as was discovered in the exploratory data analysis. The following training video also backs up this claim: 

![gif](animation1.gif)


#### Improve on the reference

From the observations above the following changes were introduced for the second training session (see [pipeline](experiments/solution/pipeline_new.config)): 

1. To avoid the huge loss gain that occurred in the first run the learning rate was halved (from 0.04 to 0.02). Also the warmup learning rate was reduced (from 0.013 to 0.005)
2. To mimic closer or further looks to the objects the `random_image_scale` data augmentation was added. This should increase the variability of the dataset by making objects appear at different distances from the car
3. To mimic the effect of direct light or changes in the dynamic range of the camera the `random_distort_color` augmentation was added. This should increase the variability of the dataset by adding more challenging detection scenarios where sudden illumination changes occur
4. to mimic the effect of driving at different hours (night, early in the morning, noon, etc) the `random_adjust_brightness` data augmentation was added. This should increase the variability of the dataset as well.

Training was conducted again over 25000 epochs and results improved significantly: 

![Screenshot from 2022-09-20 20-16-20](https://user-images.githubusercontent.com/71234974/191394256-b95e56ee-dd3e-40c0-89c9-176001e64cdf.png)

The loss still clearly improves over time and the huge jump in the reference experiment is suppressed successfully. The blue (validation) points are a little bit higher than the orange loss curves, meaning that the model may have been slightly overtrained, however they also match some local maximum in the plots meaning that this phenomenon is limited.

The learning rate as well as the loss were relatively stable at the time the experiment stopped, meaning that this is probably the best behavior that could have been obtained with these presets even is more training epochs were conducted. Note how the learning rate curve has lower values than the reference experiment


![Screenshot from 2022-09-20 19-06-10](https://user-images.githubusercontent.com/71234974/191394267-63d52b60-4318-44ce-accf-4637981355dc.png)

These were the final metrics:

    - mAP@0.5IOU: 0.089866
    - AP@0.5IOU/vehicle: 0.066611
    - AP@0.5IOU/pedestrian: 0.113122
    - AP@0.5IOU/cyclist: nan
    - Loss/localization_loss: 0.326807
    - Loss/classification_loss: 0.286569
    - Loss/regularization_loss: 0.233031
    - Loss/total_loss: 0.846408

They show a significant improvement over the reference due to the changes in the learning rate and the data augmentation techniques used. This claim is also backed up by the training video of this experiment, which hugely contrasts with the previous one:

![gifsito](animation.gif)