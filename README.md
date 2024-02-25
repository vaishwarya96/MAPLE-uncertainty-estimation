# MAPLE

Code for "Gaussian Latent Representations for Uncertainty Estimation Using Mahalanobis Distance in Deep Classifiers", ICCV Workshop 2023.
  

## Dataset

The dataset is arranged such that each class has a directory with the corresponding images placed in them. An example directory structure is shown below.

```bash
├── dataset
│   ├── train_data
│   │   ├── class1
│   │   ├── class2
...
│   │   ├── classN
│   ├── test_data
│   │   ├── class1
│   │   ├── class2
...
│   │   ├── classN

```
Each dataset is followed by a csv file containing the class name and the corresponding classification label. An example for CIFAR10 is given in `data/cifar10.csv`.

Before launching the training, please make sure that the dataset paths and the id paths (csv files) are included in the `config.py`. 


## Training

The hyperparameters and arguments needed for training the network are available in `config.py`. Depending on the dataset used, please make sure to change the respective hyperparameters. 
To launch the training, run 
```
python3 train.py
```
The code automatically splits the dataset into train and validation.

## Inference
To launch the inference, run
```
python3 mahalanobis_calculation.py
```
This calculates the Mahalanobis distance and the prediction probability for both the in distribution and out of distribution dataset, and computes the in distribution and out-of-distribution metrics.

If you use this code, please cite the following paper:

Aishwarya Venkataramanan, Assia Benbihi, Martin Laviale, Cedric Pradalier. Gaussian Latent Representations for Uncertainty Estimation using Mahalanobis Distance in Deep Classifiers. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023. p. 4488-4497.

```
@inproceedings{venkataramanan2023gaussian,
  title={Gaussian Latent Representations for Uncertainty Estimation using Mahalanobis Distance in Deep Classifiers},
  author={Venkataramanan, Aishwarya and Benbihi, Assia and Laviale, Martin and Pradalier, C{\'e}dric},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4488--4497},
  year={2023}
}
```


