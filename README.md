# Replicated GazeTR

This is the repository for the implemenation of the Gaze estimation problem reviewed in the paper "**[Appearance-based Gaze Estimation With Deep
Learning: A Review and Benchmark](https://arxiv.org/abs/2104.12668)**". The model used was GazeTR and details about it can be found in the "**[Gaze Estimation using Transformer](https://ieeexplore.ieee.org/abstract/document/9956687?casa_token=wO5ZV5h70ogAAAAA:wTP6IYD8AM9plbUoWM_SObV5sXDYH4u71GnAP4ZPDXek4bVwlxwLInXB0hzxosYbCP7ruULK)**" paper. This repository is created for the CS4240 Deep Learning course 2022/2023 and provides the instructions for the setup of the repository and describes our implementation in detail. The blog describing our approach and methodology can be found [here](https://hackmd.io/@GazeEstimationGazeTRGaze360gGrp72/BJXa7VOM2). The original details and description of the GazeTR repository can be found below.

## Requirements 

1. The GazeTR repo authors inform us to install `pytorch1.7.0`. However we had problems installing this particular version of pytorch with cuda, so we installed `torch==1.12.1+cu116` and the setup still worked. The command used was `pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116`. 

## Setup Instructions

Our setup was on a google colab instance via GCP. The instance was a `n1-highmem-2` and had 1 NVIDIA T4 gpu. The exact jupyter file with our setup is provided in the repository file.

1. First clone the GazeTR repository.
2. Download the Gaze360 dataset using the command given at the beginning of the notebook.
3. Extract the tar file containing the Gaze360 dataset files which will be the `imgs` containing head and body images for each subject in different outdoor environments. The `metadata.mat` file is needed to execute the processing code provided by the [phi-labs](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#gaze360).
4. Download the pretrained model provided by the GazeTR authors in the original repository or below this repository.
5. Set the correct paths for the `root` and `out_root` variables. These variables are the paths for the dataset and the output path for saving the processed dataset.
6. Continue executing the cells which will contain the preprocessing functions for the Gaze360 dataset used from the [phi-labs](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/#gaze360).
7. In the following directory make changes to the config yaml files for training and testing for the gaze360 dataset `GazeTR->config->train`.
8. For the training config, set the correct paths for the parameters `save` which is the directory where the trained model at each checkpoint is saved, `data` which has the paths for the processed dataset, `pretrain` which is used to define the path of the pretrained model.
9. For the testing config, set the correct path for parameter `data` which is the path for the test set used for the evaluation.
10. Feel free to change any other model parameters but the config files found in this repository contain the hyperparameter values we used.

## Training and Evaluation Instructions

1. Leave one out training was done and the following command was used: `!python /content/GazeTR/trainer/leave.py -s /content/GazeTR/config/train/config_gaze360.yaml -p 0`.
2. The original authors used a pytorch warmup optimiser for gradually increasing the learning rate, we follow their instruction. The command is `!pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git`.
3. After the training is done, we perform leave one out evaluation. The command used was `!python /content/GazeTR/tester/leave.py -s /content/GazeTR/config/train/config_gaze360.yaml -t /content/GazeTR/config/test/config_gaze360.yaml -p 0`. 
4. The angular error calculation is done in the `/tester/leave.py`. Each model checkpoint is used for the evaluation of the test set and the average angular error for each checkpoint is logged.

## Results and Discussion

A summary of the results is provided below. 

![gazetrmodelresults](https://user-images.githubusercontent.com/64498789/235206430-14359836-5196-474f-90ab-f31028e2c42c.png)

The first two rows in table 2 highlight the angular error obtained by the GazeTR paper authors on the Gaze360 dataset. Their model was trained only on ETH-XGaze dataset and then directly evaluated on the Gaze360 dataset and other datasets as well but we are only interested in the Gaze360 dataset for this replication.

Interestingly in our replication of the GazeTR-Hybrid model for the Gaze360 dataset we get an angular error of 26.51◦ which is much higher than the quoted number of 10.62◦ by the authors. Ideally the results from our replication and the authors should have been almost the same. The techniques for the setup of the GazeTR repository, the preprocessing of the dataset and the implementation were all followed from the GazeTR Github repository and no deviation was done. However the repository itself is not well documented, updated or organised. We had to frequently consult the authors phi-lab [5] website as well for the dataset preprocessing so it is possible that a crucial step for the setup or the preprocessing was missed. Another possibility is that we might have not performed the same training steps or same kind of evaluation as the authors did since there were no such details mentioned in the paper, which left a lot of room for ambiguity.

We also tuned the hyperparameter N (Number of attention heads) and evaluated the impact of pre-training through the custom and finetuned models.
Custom model - 1 had 4 attention heads and was trained from scratch on the Gaze360 dataset and evaluated on test set. The angular error obtained through this was 10.23◦ which is marginally lower than the authors published numbers which was 10.56◦.
Custom model - 2 had 8 attention heads and was trained from scratch on the Gaze360 dataset and evaluated on test set. The angular error obtained through this was 10.15◦ which is lower than the authors published numbers which was 10.62.
These results show that the training and evaluation implementation is working and seems to be correctly setup by us which makes the inconsistency in the angular error obtained by us in our replication not coming close to the authors even more peculiar. It seems like the numbers in Custom Model - 2 are a more accurate replication of the author’s original numbers but the lack of details imply that we cannot verify this.
Finally the finetuned model had 8 attention heads, used the pre-trained model as a baseline and was trained on the Gaze360 dataset and then evaluated on it’s test set acheived the best angular error of 9.05◦.


# GazeTR

We provide the code of GazeTR-Hybrid in "**Gaze Estimation using Transformer**". This work is accepted by ICPR2022.

We recommend you to use **data processing codes** provided in <a href="http://phi-ai.org/GazeHub/" target="_blank">*GazeHub*</a>.
You can direct run the method' code using the processed dataset.

<div align=center> <img src="src/overview.png"> </div>

## Requirements
We build the project with `pytorch1.7.0`.

The `warmup` is used following <a href="https://github.com/ildoonet/pytorch-gradual-warmup-lr" target="_blank">here</a>.

## Usage
### Directly use our code.

You should perform three steps to run our codes.

1. Prepare the data using our provided data processing codes.

2. Modify the `config/train/config_xx.yaml` and `config/test/config_xx.yaml`.

3. Run the commands.

To perform leave-one-person-out evaluation, you can run

```
python trainer/leave.py -s config/train/config_xx.yaml -p 0
```
Note that, this command only performs training in the `0th` person. You should modify the parameter of `-p` and repeat it.

To perform training-test evaluation, you can run

```
python trainer/total.py -s config/train/config_xx.yaml    
```

To test your model, you can run
```
python trainer/leave.py -s config/train/config_xx.yaml -t config/test/config_xx.yaml -p 0
```
or
```
python trainer/total.py -s config/train/config_xx.yaml -t config/test/config_xx.yaml
```

### Build your own project.
You can import the model in `model.py` for your own project.

We give an example. Note that, the `line 114` in `model.py` uses `.cuda()`. You should remove it if you run the model in CPU.
```
from model import Model
GazeTR = Model()

img = torch.ones(10, 3, 224 ,224).cuda()
img = {'face': img}
label = torch.ones(10, 2).cuda()

# for training
loss = GazeTR(img, label)

# for test
gaze = GazeTR(img)
```

## Pre-trained model
You can download from <a href="https://drive.google.com/file/d/1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj/view?usp=sharing" target="_blank"> google drive </a> or <a href="https://pan.baidu.com/s/1GEbjbNgXvVkisVWGtTJm7g" target="_blank"> baidu cloud disk </a> with code `1234`. 
  
This is the pre-trained model in ETH-XGaze dataset with 50 epochs and 512 batch sizes. 

## Performance
![ComparisonA](src/ComparisonA.png)

![ComparisonB](src/ComparisonB.png)

## Citation
```
@InProceedings{cheng2022gazetr,
  title={Gaze Estimation using Transformer},
  author={Yihua Cheng and Feng Lu},
  journal={International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
```

## Links to gaze estimation codes.

- A Coarse-to-fine Adaptive Network for Appearance-based Gaze Estimation, AAAI 2020 (Coming soon)
- [Gaze360: Physically Unconstrained Gaze Estimation in the Wild](https://github.com/yihuacheng/Gaze360), ICCV 2019
- [Appearance-Based Gaze Estimation Using Dilated-Convolutions](https://github.com/yihuacheng/Dilated-Net), ACCV 2019
- [Appearance-Based Gaze Estimation via Evaluation-Guided Asymmetric Regression](https://github.com/yihuacheng/ARE-GazeEstimation), ECCV 2018
- [RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments](https://github.com/yihuacheng/RT-Gene), ECCV 2018
- [MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation](https://github.com/yihuacheng/Gaze-Net), TPAMI 2017
- [It’s written all over your face: Full-face appearance-based gaze estimation](https://github.com/yihuacheng/Full-face), CVPRW 2017
- [Eye Tracking for Everyone](https://github.com/yihuacheng/Itracker), CVPR 2016
- [Appearance-Based Gaze Estimation in the Wild](https://github.com/yihuacheng/Mnist), CVPR 2015

## License
The code is under the license of [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contact 
Please email any questions or comments to yihua_c@buaa.edu.cn.
