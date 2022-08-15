### Project overview
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

Modified config file in experiment1 and experiment2 folder as shown in the table below:

Experiment0: unchanged cofig file using cosine_decay_learning_rate

```
optimizer {
	    momentum_optimizer {
	      learning_rate {
	        cosine_decay_learning_rate {
	          learning_rate_base: 0.04
	          total_steps: 2500
	          warmup_learning_rate: 0.013333
	          warmup_steps: 200
	        }
	      }
	      momentum_optimizer_value: 0.9
	    }
```
	
  
**Experiment1:** using exponential_decay_learning_rate

```
optimizer {
	    momentum_optimizer {
	      learning_rate {
	        exponential_decay_learning_rate {
	          decay_steps: 400
	          initial_learning_rate: 0.001
	        }
	      }
	      momentum_optimizer_value: 0.9
	    }
```

Training logs obtained from Temsorboard are shown below:

![improve_perf_ex1_s4_1](https://user-images.githubusercontent.com/109758200/184705983-339a62c9-5545-495c-9360-517a4948ea3c.png)

![improve_perf_ex1_s4_2](https://user-images.githubusercontent.com/109758200/184705993-f6a22cdb-9e09-4088-9d2f-ead1629609f9.png)

**Experiment2:** convert image to grey scale (randomly) and using exponential_decay_learning_rate
```
data_augmentation_options {
    random_rgb_to_gray {
      probability: 0.5
    }
```

Training logs obtained from Tensorboard are shown below:

![improve_perf_ex2_s4_1](https://user-images.githubusercontent.com/109758200/184705912-d104a093-d1d4-419a-8f5a-77e9bc97faaf.png)

![improve_perf_ex2_s4_2](https://user-images.githubusercontent.com/109758200/184705930-4cb0ce72-03c8-43c5-bdef-da2cd59c0995.png)



