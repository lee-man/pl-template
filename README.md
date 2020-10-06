# PytorchLightning Template
This repo includes the general template for deep learning training, using [PyTorch Lightning](https://github.com/PytorchLightning/pytorch-lightning) library. Use this template to start new deep learning / ML projects.

- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

## File Organization
```
`Project name`
└── main.py # The entry to run the training and evaluation.

└── model # The folder to maintain model definition.
    └── __init__.py
    └── model1.py
    └── model2.py
    └── ...
└── utils # The folder to include some useful utility functions and evalutation metrics
    └── utils.py
    └── metrics.py
└── data # The folder to maintain data module.
    └── __init__.py
    └── dataset1.py
    └── dataset2.py
    └── ...
└── conf # The folder to maintain the configuration files.
    └── setup.yaml
└── exp # The folder to maintain the logging files and model checkpointing.
    └── exp1
      └── log
      └── checkpoint

``` 

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV     
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd pl-template
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder

# run module (example: mnist as your main contribution)   
python run.py --config config/config.yml
```
### Config file
```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: 3
  latent_dim: 
    .         # Other parameters required by the model
    .
    .

```

----
## Results

...
<!-- ## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
``` -->

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
