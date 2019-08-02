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
    orcid: 0000-0002-3458-6498
    affiliation: 1
  - name: Yuki Mitsufuji
    orcid: 0000-0000-0000-0000
    affiliation: 3
affiliations:
 - name: Inria and LIRMM, University of Montpellier, France
   index: 1
 - name: Sony Europe B.V., Germany
   index: 2
 - name: Sony Corporation, Japan
   index: 3
date: 17 July 2019

bibliography: paper.bib
---

# Abstract

Music source separation is the task of decomposing music into its constitutive components e.g. yielding separated stems for the vocals, bass and drums.
Such a separation has many applications ranging from rearranging/repurposing the stems (remixing, repanning, upmixing) to full extraction (karaoke, sample creation, audio restoration).
Music separation has a long history of scientific activity as it is known to be a very challenging problem.
In recent years, deep learning based systems - for the first time - yielded high quality separations that also lead to increased commercial interest.
However, until now, no open-source implementation that achieves state-of-the-art results was available.
_Open-Unmix_ closes this gap by providing a reference implementation based on deep neural networks.
It serves two main purposes: Firstly, to accelerate academic research as the _open-unmix_ provides implementations for the most popular deep learning framework, giving researchers a flexible way to reproduce results; Secondly, we provide a pre-trained model for end users and even artists to try and use source separation.
Furthermore, we designed _Open-Unmix_ to be one core component in an open ecosystem on music separation, where we already provide open datasets, software utilities and open evaluation to fosters reproducible research as the basis of future development.

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

The most popular public repositories for deep learning are `MTG/DeepConvSep` from [@chandna17] and `f90/Wave-U-Net` from [stoller] which are more popular than all the previously mentioned separation frameworks together.

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

# Open-Unmix (UMX) (technical details)

We will now give more technical details about UMX. Fig. \ref{} shows the basic approach. During training, we learn a DNN which can be later used for separating songs.

![Block diagram of UMX](UMX_BlockDiagram.pdf)

[INSERT THE FIGURE FOR THE UMX MODEL HERE]
The design choices made for _Open-unmix_ have sought to reach two somewhat contradictory objectives. Its first aim is to have state-of-the art performance, and its second aim is to still be easily understandable, so that it could serve as a basis for research allowing improved performance in the future.

In short, _Open-unmix_ inputs mixtures in the waveform domain, transforms them through a fixed Time-Frenquency representation, before applying a series a nonlinear recurrent layers to predict the spectrogram of a target source. At the end of the chain, a postprocessing system gathers the estimates for all sources, and combines them through a multichannel Wiener filter to obtain the actual separated waveforms.

The most critical aspects of the system are the following:
* __Expert-knowledge__: while end-to-end systems that directly produce estimates in the waveform (time) domain are a promising research direction, they do not lead to state-of-the-art performance in music separation. On the contrary, systems operating in the Time-Frequency (TF) domain are still observed to significantly outperform more "modern" solutions that would bypass the expert knowledge required by TF processing.

  From the perspective of _Open unmix_, signal processing fundamentals are encapsulated in _pre_ and _post-processing_ operations. _Preprocessing_: the computation of Short-Term Fourier Transforms (STFT). _Postprocessing_: the spectrogram of the estimates is obtained by multiplying element-wise the input spectrogram through a _mask_ whose values lie between 0 and 1. This comes from the knowledge that energies of the sources roughly _add up__ to the energy of the mixture. Then, multichannel Wiener filters exploit the spectrogram estimates to produce the actual waveforms of the estimates. STFT forward and inverse transformations are used as implemented in standard libraries. Wiener filtering is used as implemented in the `sigsep.norbert` [LINK] repository.

* __Discriminative__: the system is trained to predict a separated source from the observation of its mixture with other sources. The corresponding training is done in a _discrimative_ way, i.e. through a dataset of mixtures paired with their true separated sources. These are used as groundtruth targets from which gradients are computed.

  Although alternative ways to train a separation system have emerged recently, notably through _generative_ strategies trained through adversarial cost functions, they do not lead  to comparable performance still. Even if we acknowledge that such an approach could in theory allow to scale the size of training data since it can be done in an _unpaired_ manner, we feel that this direction is still in progress and cannot be considered state of the art today.  That said, the _Open-unmix_ system can easily be extended to such generative training, and the community is much welcome to exploit it for that purpose.

* __Baseline network__: the constitutive parts of the actual deep model used in _Open-unmix_ only comprise very classical elements. Among them , we can mention:
   - _Fully connected time-distributed layers_ are used for dimension reduction and augmentation, at the input and output sides, respectively. They allow control over the number of parameters of the model and prove to be crucial for generalization.
   - _Skip connections_ are used in two ways: i/ the output to recurrent layers are augmented with their input, and this proved to help convergence. ii/ The output spectrogram is computed as an element-wise multiplication of the input. This means the system actually has to learn _how much each TF bin does belong to the target source_ and not the _actual_ value of that bin. This is _critical_ for obtaining good performance and combining the estimates given for several targets, as done in _Open-unmix_.
   - _Non linearities_ are of three kinds: i/ Rectified Linear  (ReLUE) Units allow intermediate layers to comprise nonnegative activations, which long proved effective in TF modelling. ii/ `tanh` are known to be necessary for a good training of LSTM model, notably because they avoid exploiding input and output. iii/ a `sigmoid` activation is chosen before masking, to mimmick the way legacy systems take the outputs as a _filtering_ of the input.
   - _Batch normalization_ long proved important for stable training, because it makes the different batches more similar in terms of distributions. In the case of audio where signal dynamics can be very important, this is crucial.

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
The research concerning the deep neural network architecture as well as the training process was done in close collaboration with
Stefan Uhlich and Yuki Mitsufuji from Sony Corporation.

# References
