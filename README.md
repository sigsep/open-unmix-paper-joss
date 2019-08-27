# Open-Unmix Paper

This repository combines the software contributions for _open-unmix_, a reference implementation for deep learning based music source separation.

Open-Unmix is developed for multiple frameworks (for now Pytorch, Tensorflow and NNabla). To respect the specifics of each framework we did not aim to achive identical performance for each framework. In fact, the Pytorch implementation was selected to be the _lead implementation_ as we think it currently offers the best tradeoff between simple code and easy deployment while being simple to extend for researchers.

## Software Packages

### Open-Unmix for Pytorch

* Code: [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch)
* Status: feature complete
* Tag: 1.0.0
* Pretrained models: [UMXHQ](https://zenodo.org/record/3370489) and [UMX](https://zenodo.org/record/3370486)

### Open-Unmix for NNabla

* Code: [open-unmix-nnabla](https://github.com/sigsep/open-unmix-nnabla)
* Status: model is feature complete. Extended dataset and training parameters missing
* Tag: master
* Pretrained models: not available

### Open-Unmix for Tensorflow 

* Code: [open-unmix-tensorflow](https://github.com/sigsep/open-unmix-tensorflow)
* Status: in development
* Tag: master
* Pretrained models: not available

### musdb dataset parser

A python package to parse and process the MUSDB18 dataset, the largest open access dataset for music source separation. 

* Code: [musdb](https://github.com/sigsep/sigsep-mus-db/tree/v0.3.1) 
* Tag: `v0.3.1`
* Status: released on pypi in version 0.3.1
* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3271451.svg)](https://doi.org/10.5281/zenodo.3271451)


### museval objective evaluation

* Code: [museval](https://github.com/sigsep/sigsep-mus-eval/tree/v0.3.0) 
* Tag: `v0.3.0`
* Status: released on pypi in version 0.3.0 
* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3376621.svg)](https://doi.org/10.5281/zenodo.3376621)

### norbert: wiener filter implementations

* Code: [norbert](https://github.com/sigsep/norbert/tree/v0.2.0)
* Status: released on pypi in version 0.2.0 
* Tag: `v0.2.0`
* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3269749.svg)](https://doi.org/10.5281/zenodo.3269749)

## Paper

to create the paper locally

```bash
docker run -v $PWD:/data openbases/openbases-pdf pdf
```
