# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

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
Object Detection for self-driving cars is an important task for autonomous navigation. The embedded software must be accurate and reliable for real-time computer vision image processing. In this project, we will build a Convolutional Neural Network (CNN) containing multiple layers that filter and map pixels. The image will pass through a series of convolution layers that compare small squares of input data from the image to detect 3 classes: vehicles, pedestrians, and cyclists using the Waymo Open Datset. The preprocessed data is provided in the Udacity workspace: /home/workspace/data/waymo/training_and_validation. The test data contains 3 tfrecord files the location: /home/workspace/data/waymo/test. The project explores splitting data for an efficient model and visualizing object detection with bounding boxes. The Waymo Open Dataset contains a diverse set of images to be visualized in different weather conditions and scenerios.

### Set up
Please follow the readme instructions for local setup. The Udacity Virtual Machine has all the necessary dependencies and extensions installed.

### Dataset
#### Dataset analysis
The Waymo Open Dataset was used to train a neural network model. The data within the Udacity classroom has 97 tfrecord available for training. These tfrecord files contain 1 frame per 10 seconds from a 10fps video. These images are annotated bounding boxes for 3 classes (vehicles, pedestrian, cyclists). The images from the tfrecord include:

*(Rainy)*
![obj6](https://user-images.githubusercontent.com/22205974/148433196-71afebd1-064c-4e8a-a67c-6d4c4975c1f1.PNG)
*(Sunny)*
![obj3](https://user-images.githubusercontent.com/22205974/148440152-59e9a62e-ff26-4c21-9664-fae2caa7576a.PNG)
*(Night)*
![obj2](https://user-images.githubusercontent.com/22205974/148436268-2f6b841e-640e-4d81-91df-f308e9c0f6c8.PNG)

*(Low tracked class densities)*
![obj4](https://user-images.githubusercontent.com/22205974/148440118-9c946687-fa07-45cd-837e-13b740099561.PNG)

*(Med tracked class densities)*
![obj7](https://user-images.githubusercontent.com/22205974/148440056-0b952f87-64aa-4066-be93-bfd3b3e827c3.PNG)

*(High tracked class densities)*
![obj8](https://user-images.githubusercontent.com/22205974/148439342-00718794-2b5a-4da7-8a3d-37ca2f5b5adc.PNG)


The Single Shot Detector Model is an object detection algorithm that was used to train the Waymo Open Dataset. This model detected 3 classes from the dataset: vehicles, pedestrians, and cyclist. The frequency distribution of these classes are based on the analysis of 1000 and 10,000 shuffled images in the training dataset. In 1,000 images 76% of vehicles, 24% of pedestrians and less that 1% were cyclists were tracked. This produced very few shuffled images containing cyclists.

![fd1000](https://user-images.githubusercontent.com/22205974/148423059-865c08dc-169a-41b8-9298-9fa36d5aa178.PNG)

In 10,000 images 75% of vehicles, 24% of pedestrians and 1% were cyclists were tracked. This increseased the number of cyclists tracked from the dataset.

![fd10000](https://user-images.githubusercontent.com/22205974/148423800-3efc61d5-0f4b-47ca-a478-73ad650e03e4.PNG)


#### Cross validation
97 tfrecord files were split 85:15, 82 files for training and 15 files for validation. The testing file contains 3 tfrecord file preloaded into the Udacity workspace. In order to properly train the neural network the image were shuffled for cross validation. The create_splits.py shuffles the files before splitting the dataset. Shuffling the data helps the algorithm avoid any bias due to the order of the data was assembled. These bias would cause the algorithm, to visualize patterns read from the previous images that may not be in the following images and  this would cause overfitting.

### Training
#### Reference experiment
The Single Shot Detector (SSD) Resnet 50 model was used to pretrain our dataset. Epoch help to visual overfitting. The learning rate, on the initial experiment resulted in a low learning rate

![learnrate](https://user-images.githubusercontent.com/22205974/148449119-b4fc11e9-d4bf-4dbc-8313-4e5f704d107e.png)

As the number of epochs increases the more the weights are changed in the neural network and goes from underfitting to an desired result or overfitting. 

![cap1](https://user-images.githubusercontent.com/22205974/148436976-da0253d4-05b6-4f34-a628-cda022ad1568.PNG)
![cap2](https://user-images.githubusercontent.com/22205974/148436998-034e2e3d-14c0-4b45-98f1-cb96f55bf6f3.PNG)
![cap3](https://user-images.githubusercontent.com/22205974/148437017-89dc2b3c-9215-462f-b496-9b4907b05663.PNG)

The model starts to over fit when the training set accuracy is increasing and the test set accuracy is decreasing.The models prediction, loss:0.1299, accuracy: 0.9505, val_loss: 0.4567 -val_accuracy:0.8974

![epoch](https://user-images.githubusercontent.com/22205974/148437086-3808093e-93f5-4bfe-b496-0f690cd4d809.PNG)



#### Improve on the reference
Augmenting images improves the models performance. 

The following aumentations were applied:
The image was flipped (random_horizontal_flip) This presents a mirrored image that helps to train the model to recognize objects in the opposite direction.

![augflip](https://user-images.githubusercontent.com/22205974/148433919-03d39f1a-3023-4fd6-90c0-f493956e10e7.PNG)

The image was converted into grayscale (random_rgb_to_gray) 0.02 probability. RGB images need 24 bits for processing while grayscale only needs 8bits to process. Grayscale provides faster processing times and helps with feature distinction.

![grayaug](https://user-images.githubusercontent.com/22205974/148433944-a69bbeac-a30a-46f0-9bff-3241669e0251.PNG)

The image was converted to adjust the brightness adjust by delta 0.3. Over exposure to light can make it harder for the model to distingush the objects features.

![obj5](https://user-images.githubusercontent.com/22205974/148433969-c37d2749-0604-4b47-9f79-09df4dd7865c.PNG)

The image was converted to adjust the contrast of the images to make it darker to train the model. Training the model with darker images can provide a better model for object  recognition in darker images.

![aug1](https://user-images.githubusercontent.com/22205974/148454948-1aa0089b-9e6e-413a-8d47-9a00805d60ac.PNG)



