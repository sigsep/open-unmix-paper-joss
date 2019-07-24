---
title: 'Open-Unmix - A reference implementation for audio source separation'
tags:
  - Python
  - audio
  - music
  - separation
  - deep learning
authors:
  - name: Fabian-Robert Stöter
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Stefan Uhlich
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Antoine Liutkus
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Yuki Mitsufuji
    orcid: 0000-0000-0000-0000
    affiliation: 3
affiliations:
 - name: Inria and LIRMM, Montpellier, France
   index: 1
 - name: Sony Europe B.V., Germany
   index: 2
 - name: Sony Corporate, Japan
   index: 3
date: 17 July 2019

bibliography: paper.bib
---

# Abstract

Music source separation is the task of demixing music into individual instruments, e.g., demixing songs into vocals and accompaniment.
Such a separation has many applications ranging from audio restoration to upmixing.
Until now, no open-source implementation which achieves state-of-the-art results was available.
'Open-Unmix' closes this gap by providing a open reference implementation which achieves state-of-the-art performance.
Our design goal was to develop code that is targeted at researchers as well as users.
With 'Open-Unmix' we want to give the community the missing piece to do their own research and that it becomes the basis for future research.

# Introduction

- open up to deep learning community,
- Explain the story of music separation and how it got popular in research
- list applications
- Explain how and when deep neural network based music separation outperformed existing methods.
- Now, there are also commercial systems based on machine learning were released _Audionamix XTRAX STEMS_ or _IZOTOPE RX 7_  or _AudioSourceRE_

- context, vision, SiSEC 2007, compare performance community, promote research.
- datasets, tools.
- reference baseline.
- what is a baseline? Many implementations from legacy methods. Modern methods as state-of-the-art. A few open implementations but none of them art state-of-the-art.
- what do we call baseline?

- where is research to be expected:
  - why  do we need domain knowledge?
  - crucial part for domain knowledge in audio?
  - domain knowledge, where is the knowledge?
  - representations (needs tested model)
  - model (needs tested pre-and postprocess)

##  Related libraries/work

- In the open there are several source separation libraries that aimed to implement a bunch of methods

### openBlissart

- presented in [@weninger11]
- represents state-of-the-art from 2011 (NMF)

### Flexible Audio Source SeparationToolbox(FASST) 

- Presented in [@salaun12]
- Written in MATLAB and C++
- outdated methods

### untwist

- presented in [@roma16]
- doesn't seem ready to use
- outdated methods

### nussl

- presented in [@manilow18]
- complex and focussed on signal processing methods instead of DNNs
- built-in interfaces for common evaluation metrics, data sets, or loading pre-trained models.

While these frameworks attracted some users (github stars – nussl: 196 stars, untwist: 95, pyfasst 79, openblissart 77), they didn't get much traction compared to specific deep learning methods.

The most popular public repositories for deep learning are `MTG/DeepConvSep` from [@chandna17] and `f90/Wave-U-Net` from [@stoeller] which are more popular than all the previously mentioned separation frameworks together.

Also mention because they are very popular:

- [https://github.com/andabi/music-source-separation] Tensorflow
- [https://github.com/posenhuang/deeplearningsourceseparation]  Matlab

## The gap

- An open-source separation method that performs as good as state-of-the-art was missing now for >4 years
- Many users currently cannot assess if a method performs as good as state-of-the-art thus a true open baseline is missing.
- Result is that it is often believed that eg. spectrogram u-net outperforms older methods (=not true).
- today many new users approach music separation from the ML perspective but they lack domain knowledge and therefore might produce subpar results (as this is still important) 
- Many methods/researchers face difficulties in pre and post-processing, since we are experienced researchers in this area we put our combined domain knowledge into _open-unmix_, its data loading and post-processing
- these ML researchers are not looking for a general framework on source separation but a SOTA method that is easy to extend.
- Deep learning is field with fast progress: techniques will probably stay for a while, but frameworks will rapidly evovlve
- _Open-unmix_ therefore is developed in parallel to cover the most number of users... tensorflow/keras, pytorch, nnabla
- The pytorch version will serve as the reference version due its simplicity and easyness to extend the code
- the tensorflow version will be release later when TF 2.0 is stable. 
- version for nnabla will be close the pytorch code "example" and will be released together with the tensorfloe version.

# Open-Unmix (technical details)

- how does it work
- data loading -> data sampling -> preprocessing -> model/training -> inference -> wiener filter

## why LSTM?
- Open-unmix can easily be extended to include different models. Currently implemented: LSTM network
- Recent Research on End-To-End models are tempting, because they can get away with domain knowledge typical required produce good results
- However, none of the modern networks design produced state-of-the-art results (e.g. https://github.com/francesclluis/source-separation-wavenet based on [https://arxiv.org/abs/1810.12187])
- Even worse, methods that use proven network architecture such as RNNs often didn't match state-of-art results

## The source separation community

- Open-unmix is part of a whole ecosystem of software, datasets and online resources: the `sigsep` community
- we provide MUSDB18 and MUSDB18-HQ, the largest freely available dataset, this comes with a complete toolchain to easily parse and read the dataset such as [musdb] and [mus]
- [museval], is mostly used evaluation package for source separation
- we also are the organizers of the largest source separation evaluation campaign
- in this campaign we noticed: previous state-of-the-art systems could not be matched by newer systems (e.g. UHL2)

## Features

- Modular, easy to extend, using framework agnostic
- pytorch implementation based on the famous MNIST example. 
- reproducible code
- includes unit tests and regression tests.

## Results

compare and list link to demo website

# Contributions

Open-Unmix was developed by Fabian-Robert Stöter and Antoine Liutkus at Inria Montpellier.
The design of the deep neural network architecture was done in close collaboration with
Stefan Uhlich and Yuki Mitsufuji from Sony Corporation.

# References
