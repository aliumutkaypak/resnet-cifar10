# resnet-cifar10
NYU Tandon Deep Learning Mini-Project Repository\
This repo adapts the ResNet class from https://github.com/kuangliu/pytorch-cifar.

### Group Members
Ali Umut Kaypak | Aashray Pola | Mohnish Bangaru \
{ak10531,ap8130,mb9628}@nyu.edu

### Repo Structure
Training code is in /src/resnet_training.ipynb. It takes the model and training hyperparameters, trains the model, creates the loss and accuracy curves and saves the output csv file for the unlabeled dataset. 

Inference code is in /src/resnet_inference.ipynb. It takes the model checkpoint path, model hyperparameters (model hyperparameters must match with the corresponding model checkpoint and they can be found in models/model_{i}/hyperparameters.txt for each model). It creates the output csv file for the unlabeled dataset and outputs test, train, validation accuracies. 

Model checkpoints are saved in /models/model_{i}/model.pth for model i. Its hyperparameters can be found in /models/model_{i}/hyperparamters.txt. These hyperparameters should be used in the inference notebook to see model_{i} results.

### Model details
All the models are trained using SGD with 0.1 learning rate with 0.9 momentum and $5\times10^{-4}$ weight decay. Cosine Annealing lr schedular is used for lr scheduling. 2 different set of augmentation strategies are tried:\
Set1: RandomCrop(32, padding=4) + RandomHorizontalFlip() + RandomRotation(15) +Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) \
Set2: RandomCrop(32, padding=4) + RandomHorizontalFlip() + [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf) + Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

| Model Name | Augment Strategy | Dropout Rate |    # of  channels    | # of blocks  in layers | # of Params | Test  Accuracy |
|:----------:|:----------------:|:------------:|:--------------------:|:----------------------:|:-----------:|:--------------:|
|   model_4  |       Set 2      |      0.0     | [32, 64, 128, 256]   |      [3, 4,  4, 3]     |  4,747,914  |     95.41%     |
|   model_5  |       Set 2      |      0.0     | [32, 64, 128, 256]   |      [4, 5,  4, 3]     |  4,840,458  |     95.23%     |
|   model_1  |       Set 2      |      0.0     | [64, 128,  256, 512] |      [2, 1,  1, 1]     |  4,977,226  |     95.06%     |
|   model_2  |       Set 1      |      0.0     | [64, 128,  256, 512] |      [2, 1,  1, 1]     |  4,977,226  |     93.90%     |
|   model_3  |       Set 2      |      0.5     | [64, 128,  256, 512] |      [2, 1,  1, 1]     |  4,977,226  |     90.71%     |