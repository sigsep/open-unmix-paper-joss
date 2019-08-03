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

# Background

<!-- - Explain the story of music separation and how it got popular in research -->
Separating music signals is a problem researchers have been fascinated about for over 50 years. This is partly due to the fact that, mathematically, there exist no closed-form solution for the typical problem of many sources captured by a single microphone.
As the problem difficult to solve on general signal, researchers instead focussed on strong assumptions about the way the signal were recorded and mixed. A large number of these methods are centered around "traditional" signal processing methods, for a more detailed overview see [@rafii17] and [@cano19].
<!-- - Explain how and when deep neural network based music separation outperformed existing methods. -->
The history of music separation is closely tied to the availability of data.
Many of these classical signal processing based methods were hand-crafted and tuned to a small number of music recordings [@sisec10, @sisec11, @sisec13].
Systematic objective evaluation of these methods, however, was hardly feasible as freely available dataset did not exist at that time.
In fact, for a meaningful evaluation, the ground truth separated stems are necessary, however, these are considered precious artifacts of the music mixing and mastering and therefore usually not available for sale.
Furthermore, any commercial music is well known to be subject to copyright protection laws and therefore generally not available for re-distribution.
Nonetheless, in the past five years, freely available datasets were released that enabled the development of data-driven methods.
This quickly lead to a large performance boost as it was seen in other audio tasks such as automatic speech recognition (ASR).
In fact, in 2016 the speech recognition community had access to datasets with more than 10.000 hours of speech [deepspeech2] which boosted the performance significantly.
At the same time, the _MUSDB_ dataset was released [@rafii17] which comprises 150 full length music tracks – a total of 10 hours of music.
Up to date, this is still the largest freely available dataset for source separation.
Nonetheless even with this small amount of data, deep neural networks(DNNs) were not only been successfully used for music separation but they are now setting the state-of-the art in this domain as can be seen by the results of the community-based signal separation evaluation campaign(SiSEC) [@sisec15, @sisec16, @sisec18].

In these challenges, the proposed systems are compared to other methods as well as a number of oracle methods [cite_oracles] to indicate possible upper bounds. On the other end, some of the classical signal processing based methods, known to be _reliable_, _fast_, _simple to use_, were treated as baseline methods.
In fact, in the past, a number of systems and libraries were explicitly being designed with these aspects in mind.
While today there exist also an number commercial systems such as _Audionamix XTRAX STEMS_, _IZOTOPE RX 7_ or _AudioSourceRE_, that target end-users, we considered only tools that are available as open source software, suitable for research.

<!-- list of methods -->
The first publicly available software for source separation was presented _openBlissart_ in 2011 [@weninger11]. It is based on C++ and represents the class of systems that are based on non-negative matrix factorization (NMF). In 2012, the _Flexible Audio Source Separation Toolbox (FASST)_ was presented in [@salaun12]. It was written in MATLAB and C++. It was also based on NMF methods but additionally also includes other model based methods.
In 2016, the _untwist_ libraries was proposed in [@roma16]. It comprises a number of methods, ranging from classical signal processing based methods to also feed forward neural networks. The library in written in Python 2.7 and unfortunately was not updated since 2 years and many methods are not ready to use.
_Nussl_ is a very recent framework, presented in [@manilow18]. It includes a large number of methods and generally is focussed on classical signal processing methods instead of machine learning. It has built-in interfaces for common evaluation metrics, data sets.

<!-- ## The gap -->
The main problem with all of these available tools is that they do not deliver state-of-the-art results. In fact, no open system exist today that matches the performance of the state-of-the-art system proposed more than 4 years [uhlich14].
We believe that the lack of such a baseline has a serious negative impact on future research on source separation.
In fact, many new methods that were published in the last years, usually compared to their own baseline, thus showing relative instead of absolute performance so that other researchers cannot assess if a method performs as good as state-of-the-art.
Also the lack of a common baseline methods miss-guides new researchers and students that enter the field of music separation. The result of this can be observed by looking at the popularity of the above mentioned music separation frameworks on github: all of the frameworks combined are less popular than specific new deep learning based methods that shared their code accompanying their paper (nussl: 196 stars, untwist: 95, pyfasst 79, openblissart 77) such as `MTG/DeepConvSep` from [@chandna17] and `f90/Wave-U-Net` from [stoller].

# Open-Unmix 

## Design Choices / Requirements

today many new users approach music separation from the ML perspective but they lack domain knowledge and therefore might produce subpar results.
- make people do less misstakes on pre- and postprocessing
- deliver state-of-the-art results
- part of sigsep community
- community driven and maintained
- closely tight to dataset / MUSDB18
- simple
- hackable (MNIST like)

## Ready for future research

- Many methods/researchers face difficulties in pre and post-processing, since we are experienced researchers in this area we put our combined domain knowledge into _open-unmix_, its data loading and post-processing
- these ML researchers are not looking for a general framework on source separation but a SOTA method that is easy to extend.

Designed so that researchers can easily hack the code to implement

- new representations (needs tested model)
- new architectures (needs tested pre-and postprocess)


# Open-Unmix (UMX) (technical details)

![](https://docs.google.com/drawings/d/e/2PACX-1vTPoQiPwmdfET4pZhue1RvG7oEUJz7eUeQvCu6vzYeKRwHl6by4RRTnphImSKM0k5KXw9rZ1iIFnpGW/pub?w=959&h=308)

- how does it work
- data loading -> data sampling -> preprocessing -> model/training -> inference -> wiener filter

We will now give more technical details about UMX. Fig. \ref{} shows the basic approach. During training, we learn a DNN which can be later used for separating songs.

![Block diagram of UMX](UMX_BlockDiagram.pdf)

[INSERT THE FIGURE FOR THE UMX MODEL HERE]
The design choices made for _Open-unmix_ have sought to reach two somewhat contradictory objectives. Its first aim is to have state-of-the art performance, and its second aim is to still be easily understandable, so that it could serve as a basis for research allowing improved performance in the future.

In short, _Open-unmix_ inputs mixtures in the waveform domain, transforms them through a fixed Time-Frenquency representation, before applying a series a nonlinear recurrent layers to predict the spectrogram of a target source. At the end of the chain, a postprocessing system gathers the estimates for all sources, and combines them through a multichannel Wiener filter to obtain the actual separated waveforms.

The most critical aspects of the system are the following:
* __Expert-knowledge__: while end-to-end systems that directly produce estimates in the waveform (time) domain are a promising research direction, they do not lead to state-of-the-art performance in music separation. On the contrary, systems operating in the Time-Frequency (TF) domain are still observed to significantly outperform more "modern" solutions that would bypass the expert knowledge required by TF processing.

  From the perspective of _Open unmix_, signal processing fundamentals are encapsulated in _pre_ and _post-processing_ operations. _Preprocessing_: the computation of Short-Term Fourier Transforms (STFT). _Postprocessing_: the spectrogram of the estimates is obtained by multiplying element-wise the input spectrogram through a _mask_ whose values lie between 0 and 1. This comes from the knowledge that energies of the sources roughly _add up__ to the energy of the mixture. Then, multichannel Wiener filters exploit the spectrogram estimates to produce the actual waveforms of the estimates. STFT forward and inverse transformations are used as implemented in standard libraries. Wiener filtering is used as implemented in the `sigsep.norbert` [LINK] repository.

* __Discriminative__: the system is trained to predict a separated source from the observation of its mixture with other sources. The corresponding training is done in a _discrimative_ way, i.e. through a dataset of mixtures paired with their true separated sources. These are used as ground truth targets from which gradients are computed.

  Although alternative ways to train a separation system have emerged recently, notably through _generative_ strategies trained through adversarial cost functions, they do not lead  to comparable performance still. Even if we acknowledge that such an approach could in theory allow to scale the size of training data since it can be done in an _unpaired_ manner, we feel that this direction is still in progress and cannot be considered state of the art today.  That said, the _Open-unmix_ system can easily be extended to such generative training, and the community is much welcome to exploit it for that purpose.

* __Baseline network__: the constitutive parts of the actual deep model used in _Open-unmix_ only comprise very classical elements. Among them , we can mention:
   - _Fully connected time-distributed layers_ are used for dimension reduction and augmentation, at the input and output sides, respectively. They allow control over the number of parameters of the model and prove to be crucial for generalization.
   - _Skip connections_ are used in two ways: i/ the output to recurrent layers are augmented with their input, and this proved to help convergence. ii/ The output spectrogram is computed as an element-wise multiplication of the input. This means the system actually has to learn _how much each TF bin does belong to the target source_ and not the _actual_ value of that bin. This is _critical_ for obtaining good performance and combining the estimates given for several targets, as done in _Open-unmix_.
   - _Non linearities_ are of three kinds: i/ Rectified Linear  (ReLUE) Units allow intermediate layers to comprise nonnegative activations, which long proved effective in TF modelling. ii/ `tanh` are known to be necessary for a good training of LSTM model, notably because they avoid exploiding input and output. iii/ a `sigmoid` activation is chosen before masking, to mimmick the way legacy systems take the outputs as a _filtering_ of the input.
   - _Batch normalization_ long proved important for stable training, because it makes the different batches more similar in terms of distributions. In the case of audio where signal dynamics can be very important, this is crucial.

* __Data loading__:

   - efficient sampling
   - good data loading
   - fast 
   - essential augmentations

## why LSTM?

[@Hochreiter97]

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

## Implementation Details

- _Open-unmix_ is developed in parallel to cover the most number of users... tensorflow/keras, pytorch, nnabla
- The pytorch version will serve as the reference version due its simplicity and easyness to extend the code
- the tensorflow version will be release later when TF 2.0 is stable.
- version for nnabla will be close the pytorch code "example" and will be released together with the tensorfloe version.

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
