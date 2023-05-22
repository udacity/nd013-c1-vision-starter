# Project Overview
The objective of this project is performing data exploratory analysis and train a single-shot detection model to detect objects in an urban environment using the Waymo dataset.
# Setup
The Jupyter Notebook QuanBach-Exploratory Data Analysis is the deliverable for the Exploatory Data Analysis Part
The Jupyter Notebook QuanBach_Explore_augmentations is the implementation for data augumentation
# Dataset
## Data Analysis
After implemented the draw bounding boxes for the ground truth classes, I have learnt that the ground truth classes are not 1: vehicles, 2: pedestrian, and 3: cyclist as the initial assumption but the correct classes are 1: vehicles, 2: pedestrian, and 4: cyclist.

![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/8fcae009-5529-44af-a659-664636b18b17)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/3b4a63a7-e284-46f7-84cd-35e02f301f53)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/aee916e4-e940-424e-9f4a-13fd69b0a73b)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/54c1d161-c734-4160-a6d1-e9983c3e1ebc)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/fc7dbb35-f8c0-4d34-9f7e-9f31cb6a9bfd)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/18595a1e-7f42-4645-a4f1-111c0c1dbd99)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/49385c3c-e7f1-4231-893f-5fa53601bb7d)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/54adb2cf-f6f1-4226-bda0-d7106a439246)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/65281a2b-c541-4dcf-a58f-2bb17bd36030)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/67e1db72-35d0-4055-905f-4258b78e5998)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/65dab4dc-e692-42d6-b46a-1a8ad7534497)

From the display sample images, most of the objects appear to be occluded. Therefore, adding occlusions as an data augmentation method would be beneficial for better performance.
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/73a7f55c-72d9-49a9-83c6-a4d16709839d)
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/c741da16-c3cd-407f-bf85-abe4a7262eab)

## Additional EDA
Further data exploration and analysis from sampling 100,000 batches of the dataset indicate that cyclists appear significantly less in the training dataset compared to the other two. This would result in the model having problem detecting cyclists.
1: vehicles, 2: pedestrian, and 4: cyclist.

{2: 1044151, 1: 2501861, 4: 28288}

![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/e4ccc0dd-0eba-44fb-87e5-d6c71f6694cf)

# Training
## Reference Experiment
The Reference Experiment results in poor performance with the Total Loss never got below 20. 
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/cf95cc28-39d6-4bb7-84e2-3cd4a1fe52e6)

The Normalized Loss and Regulation Loss remain low for half/one-quarter of the data respectively then increase sharply around 30 and become stagnation also.
![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/fd4c4b96-7d9e-4928-99f6-0e0218be31ef)

The Classification Loss and Localization Loss oscillated and never become logarithmic.  

![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/67a7a650-36ec-4e57-805f-4b70a5c71e8c)

## Improve On Reference Experiment
I have made multiple changes in the training configurations to improve the training. The first change is the optimizer. Instead of using the momentum optimizer I have chosen to use ADAM optimizer because Adam doesn’t roll so fast after jumping over the minimum but decreasing the velocity a little bit for carefully search. I also have different learning rates at different step in the training; details can be found in the pipeline_new.config. Furthermore, I have included two more data augmentation methods: random black patches and random color distort. These two data augmentations are chosen due to the discovery that the objects in the training data usually being occluded; and for some sample data, the rain has distort the color of the objects. The results of the improvement can be found in the Loss graphs below (with highlighted ‘improve on reference’ to indicate the new experiment and the others were the initial experiment losses) 

<img width="358" alt="image" src="https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/9373d8a7-24bd-4a5e-98c6-f279a6c96774">

![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/54e26186-9faf-414c-b38d-aea1e55669fc)

![image](https://github.com/ghost-qb/SDE_Object_Detection/assets/58492405/0d97ce33-c526-4be6-85c5-7e4eb0d228fb)
