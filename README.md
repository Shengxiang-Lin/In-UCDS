<h1 align="center">
In-UCDS    
</h1>

Recommender systems are typically biased toward a small group of users, leading to severe unfairness in recommendation performance, i.e., User-Oriented Fairness (UOF) issue. The existing research on UOF is limited and fails to deal with the root cause of the UOF issue: the learning process between advantaged and disadvantaged users is unfair. To tackle this issue, we propose an In-processingUser Constrained Dominant Sets (In-UCDS) framework, which is a general framework that can be applied to any backbone recommendation model to achieve user-oriented fairness. We split In-UCDS into two stages, i.e., the UCDS modeling stage and thein-processing training stage. In the UCDS modeling stage, for each disadvantaged user, we extract a constrained dominant set (a user cluster) containing some advantaged users that are similar to it. In the in-processing training stage, we move the representations of disadvantaged users closer to their corresponding cluster by calculating a fairness loss. By combining the fairness loss with the original backbone model loss, we address the UOF issue and maintain the overall recommendation performance simultaneously. Comprehensive experiments on three real-world datasets demonstrate that In-UCDS outperforms the state-of-the-art methods, leading to a fairer model with better overall recommendation performance.

[In-processing User Constrained Dominant Sets for User-Oriented Fairness in Recommender Systems](https://dl.acm.org/doi/abs/10.1145/3581783.3613831)  
Zhongxuan Han, Chaochao Chen, Xiaolin Zheng, Weiming Liu, Jun Wang, Wenjie Cheng, Yuyuan Li  
MM '23: Proceedings of the 31st ACM International Conference on Multimedia

## Prerequisites 
* python==3.9.21
* numpy==1.23.0
* torch==1.13.0
* tqdm==4.64.0
* scikit-learn==0.24.2
* pandas==1.5.0
* matplotlib==3.5.1

## Model Folders
The model folders include `S-DRO`, `In-Naive`, `In-UCDS`, `UFR`, and `original`, each of which implements different recommendation models and fairness treatments. 
### `S-DRO` folder 
This folder implements the S-DRO method for solving fairness problems in recommender systems.
- `config.py` : configuration file, used to set various parameters for model training, such as random seed, number of training rounds, dataset, etc.
- `main.py`: the main program file, which contains the training, tuning and testing logic of the model. In the testing phase, the optimal model parameters are loaded, the overall, active and inactive users are evaluated, and the user fairness metric (UGF) is calculated.
- `models/`: stores the definition files for the models, including `NeuMF`, `PMF` and `VAECF` models.
    - `NeuMF.py`: implements the Neural Matrix Factorization (NeuMF) model, which combines the advantages of matrix factorization and multilayer perceptual machines.
    - `PMF.py`: implements the Probabilistic Matrix Factorization (PMF) model, which models potential features of users and items through probability distributions.
    - `VAECF.py`: implements the Variational Autoencoder for Collaborative Filtering (VAECF) model to learn the latent representations of users and items using variational autoencoders.
- `myloss.py`: customized loss function file to compute the loss of the model.
- `sigdatasets.py`: dataset processing file, responsible for loading and processing datasets.

### `In-Naive` folder 
This folder implements a simple fairness treatment that may be used as a baseline model for comparison.
- `config.py` : the same configuration file as in `S-DRO` that sets the model training parameters.
- `main.py` : main program file containing the training, tuning and testing logic for the model. The testing phase is similar to `S-DRO` and evaluates the performance and fairness of the model on different user groups.
- `models/`: definition file that holds the models, currently contains only `NeuMF` models.
- `myloss.py`: custom loss function file.
- `sigdatasets.py` : dataset processing file.

### `In-UCDS` folder 
This folder implements the In-UCDS (In-processing User-Centric Disparity Smoothing) framework for solving the problem of user-oriented fairness in recommender systems.
- `config.py`: configuration file that sets the training parameters for the In-UCDS framework.
- `main.py`: main program file, responsible for the training of the In-UCDS framework. During the testing phase, the parameters of both the original and fair models are loaded to compare the performance and fairness of both on overall, active and inactive users.
- `models/`: holds the definition files for the models, including the `NeuMF` and `VAECF` models.
- `myloss.py`: file for custom loss functions.
- `sigdatasets.py`: dataset processing files.
- `test.py`: test program file for evaluating the performance of the trained model. Similar to the test logic in `main.py`, but can be run separately.
- `ucds.py` : the core code that implements the In-UCDS framework, including the UCDS modeling phase and the in-processing training phase.

### `UFR` folder 
This folder implements the UFR (User Fairness Regularization) model, which improves the fairness of the recommender system by regularizing terms.
- `UFR.py` : Code file that implements the UFR model.
- `config.py` : configuration file to set the training parameters for the UFR model.
- `data_loader.py`: data loader, responsible for loading and processing the dataset.
- `main.py`: the main program file, which contains the training, tuning and testing logic of the UFR model. The testing phase is similar to the other folders and evaluates the performance and fairness of the model.
- `models/`: definition file that holds the models, including `NeuMF` and `VAECF` models.
- `myloss.py`: custom loss function file.
- `sigdatasets.py`: dataset processing file.
- `utils/`: folder where some utility functions are stored.

### `original` folder 
This folder implements the original recommendation model, regardless of fairness issues, and serves as a baseline model for comparison.
- `config.py` : configuration file to set training parameters for the original model.
- `main.py`: the main program file, containing the training, tuning and testing logic for the original model. The testing phase evaluates the performance of the original model on different user groups.
- `models/`: definition file that holds the models, including `NeuMF`, `PMF` and `VAECF` models.
- `myloss.py`: file for custom loss functions.
- `sigdatasets.py`: dataset processing files.    
     
## Getting Started
1. Clone this repo:  
```
   git clone https://github.com/Shengxiang-Lin/In-UCDS.git  
   cd In-UCDS
``` 
2. Create a Virtual Environment  
```
   conda create -n UCDS python=3.9.21
   conda activate UCDS
```
3. Install all the dependencies  
```
   pip install -r requirements.txt
```

## Dataset
The complete source code, including datasets, is available on both [Baidu Netdisk](https://pan.baidu.com/s/1zNkoOw2R2PoFcepRvMIVzw?pwd=ppmb) and [Google Drive](https://drive.google.com/drive/folders/1L1tTwiRsuXU_pkDguPJA6BD_5IzJB0Fq?usp=sharing).
There's no need to download it though, it's already in the code file

## Training 
Run the following command to start training:
```
python main.py
```

## Testing
The trained models (weights) are available at [here](https://drive.google.com/drive/folders/17D5sp5mdNOIgRAXohBHU5fBg6-T4BMlk?usp=sharing).Download it, replace the `logs` folder, select the recommendation model and dataset you want to evaluate, and run the following code:
```
python test.py
```

## Note
This is based on copied code from the [official GitHub repository](https://github.com/hahhacx/In-UCDS/), fixing some issues in the original code (e.g. file import, data handling, etc.) and verifying its usability.
  
## Citation
If you find this code/models useful, please consider citing this paper: 
```
@inproceedings{han2023ucds,
author = {Han, Zhongxuan and Chen, Chaochao and Zheng, Xiaolin and Liu, Weiming and Wang, Jun and Cheng, Wenjie and Li, Yuyuan},
title = {In-processing User Constrained Dominant Sets for User-Oriented Fairness in Recommender Systems},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {6190–6201},
year = {2023},
series = {MM '23}
}
```
