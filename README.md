<!--
 * @Copyright (c) vaishnavmohit All Rights Reserved.
 * @Author         : Mohit Vaishnav
 * @Github         : https://github.com/vaishnavmohit
 -->

# Setting up environment and installing the libraries  
```
module load python/3.7.4
python3 -m venv ../../env/visreason
source ../../env/visreason/bin/activate
python -m pip install -r requirements.txt  
```


## Usage

---

To train a resnet50 model on ImageNet following [this](https://github.com/pytorch/examples/blob/master/imagenet/main.py) simple setup, use:

```
$ python train.py
```

To test the model that you have trained on a specific datetime:

```
$ python inference.py test.checkpoint=data/runs/YYYY-MM-DD_HH-MM-SS
```

1. Read carefully with some arguments in `train_net.py`
2. Run: `sh /scripts/train_net.sh`

## For hyperparameter search:
Use the instruction mentioned in [this](https://github.com/ashleve/lightning-hydra-template) link.

### Running the code:
'''
./scripts/run_optuna.sh config-file -m *required parameters 

-m is used for multiple runs

optuna config files start with the prefix optuna in 'config/'

'''

## For maximizing F1 or Val_acc_1:

`baseline/models/base.py: L449: 'monitor': 'val_acc_top1'`

`baseline/train_optuner/train_v3.py: L165`

`config/callback/default_optuner.yaml: model_checkpoint`


#### Originally adapted from:

`azouaoui-cv/resnet50-imagenet-baseline`