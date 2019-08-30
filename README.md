# Open-Unmix Paper

This repository combines the software contributions for _open-unmix_, a reference implementation for deep learning based music source separation.

We choose _PyTorch_ to serve as a reference implementation for this submission due to its balance between simplicity and modularity. Furthermore, we already ported the core model to [NNabla](https://github.com/sigsep/open-unmix-nnabla) and plan to release a port for Tensorflow 2.0, once the framework is released. Note that the ports will not include pre-trained models as we cannot make sure the ports would yield identical results, thus leaving a single baseline model for researchers to compare with

## Software Packages

### Open-Unmix for Pytorch

* Code: [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch)
* Status: feature complete
* Tag: 1.0.0
* Pretrained models: [UMXHQ](https://zenodo.org/record/3370489) and [UMX](https://zenodo.org/record/3370486)
* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3382104.svg)](https://doi.org/10.5281/zenodo.3382104)

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
