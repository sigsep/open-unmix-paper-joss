---
title: 'Open-Unmix - A Reference Implementation for Music Source Separation'
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
    orcid: 0000-0003-3158-4945
    affiliation: 2
  - name: Antoine Liutkus
    orcid: 0000-0002-3458-6498
    affiliation: 1
  - name: Yuki Mitsufuji
    orcid: 0000-0002-6806-6140
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

# Summary

Music source separation is the task of decomposing music into its constitutive components, e.g., yielding separated stems for the vocals, bass, and drums.
Such a separation has many applications ranging from rearranging/repurposing the stems (remixing, repanning, upmixing) to full extraction (karaoke, sample creation, audio restoration).
Music separation has a long history of scientific activity as it is known to be a very challenging problem.
In recent years, deep learning-based systems - for the first time - yielded high-quality separations that also lead to increased commercial interest.
However, until now, no open-source implementation that achieves state-of-the-art results is available.
_Open-Unmix_ closes this gap by providing a reference implementation based on deep neural networks.
It serves two main purposes. Firstly, to accelerate academic research as _Open-Unmix_ provides implementations for the most popular deep learning frameworks, giving researchers a flexible way to reproduce results. Secondly, we provide a pre-trained model for end users and even artists to try and use source separation.
Furthermore, we designed _Open-Unmix_ to be one core component in an open ecosystem on music separation, where we already provide open datasets, software utilities, and open evaluation to foster reproducible research as the basis of future development.

# Background

Music separation is a problem which has fascinated researchers for over 50 years. This is partly because, mathematically, there exists no closed-form solution when many sources (instruments) are recorded in a mono or stereo signal.
To address the problem, researchers exploited additional knowledge about the way the signals were recorded and mixed. A large number of these methods are centered around "classical" signal processing methods. For a more detailed overview see [@rafii17] and [@cano19].
<!-- - Explain how and when deep neural network-based music separation outperformed existing methods. -->
Many of these methods were hand-crafted and tuned to a small number of music recordings [@sisec10; @sisec11; @sisec13].
Systematic objective evaluation of these methods, however, was hardly feasible as freely available datasets did not exist at that time.
In fact, for a meaningful evaluation, the ground truth separated stems are necessary.
However, because commercial music is usually subject to copyright protection, and the separated stems are considered to be valuable assets in the music recording industry, they are usually unavailable.

Nonetheless, thanks to some artists who choose licenses like Creative Commons, that allow sharing of the stems, freely available datasets were released in the past five years and have enabled the development of data-driven methods.
Since then, progress in performance has been closely linked to the availability of more data that allowed the use of machine-learning-based methods.
This led to a large performance boost similar to other audio tasks such as automatic speech recognition (ASR) where a large amount of data was available.
In fact, in 2016 the speech recognition community had access to datasets with more than 10000 hours of speech [@amodei16].
In contrast, at the same time, the _MUSDB18_ dataset was released [@rafii17] which comprises 150 full-length music tracks – a total of just 10 hours of music.
To date, this is still the largest freely available dataset for source separation.
Nonetheless, even with this small amount of data, deep neural networks (DNNs) were not only successfully used for music separation but they are now setting the state-of-the-art in this domain as can be seen by the results of the community-based signal separation evaluation campaign (SiSEC) [@sisec15; @sisec16; @sisec18]. In these challenges, the proposed systems are compared to other methods. Among the systems under test, classical signal processing based methods were clearly outperformed by machine learning methods. However they were still useful as a _fast_ and often _simple to understand_ baseline.

In the following, we will describe a number of these reference implementations for source separation.
While there are some commercial systems available, such as _Audionamix XTRAX STEMS_, _IZOTOPE RX 7_ or _AudioSourceRE_, we only considered tools that are available as open-source software, and are suitable for research.

<!-- list of methods -->
The first publicly available software for source separation was _openBlissart_, released in 2011 [@weninger11]. It is written in C++ and accounts for the class of systems that are based on non-negative matrix factorization (NMF). In 2012, the _Flexible Audio Source Separation Toolbox (FASST)_ was presented in [@ozerov2011general; @salaun14]. It is written in MATLAB/C++ and is also based on NMF methods, but also includes other model-based methods.
In 2016, the _untwist_ library was proposed in [@roma16]. It comprises several methods, ranging from classical signal-processing-based methods to feed-forward neural networks. The library is written in Python 2.7. Unfortunately, it has not been updated since 2017 and many of its methods are not subjected to automated testing.
_Nussl_ is a very recent framework, presented in [@manilow18]. It includes a large number of methods and generally focuses on classical signal processing methods rather than machine-learning-based techniques. It has built-in interfaces for common evaluation metrics and data sets. The library offers great modularity and a good level of abstraction. However, this also means that it is challenging for beginners who might only want to focus on changing the machine learning parts of the techniques.

<!-- ## The gap -->
The main problem with these implementations is that they do not deliver state-of-the-art results. No open-source system is available today that matches the performance of the best system proposed more than four years ago by [@uhlich15].
We believe that the lack of such a baseline has a serious negative impact on future research on source separation.
Many new methods that were published in the last few years are usually compared to their own baseline implementations, thus showing relative instead of absolute performance gains, so that other researchers cannot assess if a method performs as well as state-of-the-art.
Also, the lack of a common reference for the community potentially misguides young researchers and students who enter the field of music separation. The result of this can be observed by looking at the popularity of the above-mentioned music separation frameworks on GitHub: all of the frameworks mentioned above, combined, are less popular than two recent deep learning papers that were accompanied by code such as `MTG/DeepConvSep` from [@chandna17] and `f90/Wave-U-Net` from [@stoller18]. Thus, users might be confused regarding which of these implementations can be considered state-of-the-art.

# Open-Unmix

We propose to close this gap with _Open-Unmix_, which applies machine learning to the specific tasks of music separation.
With the rise of simple to use machine learning frameworks such as _Pytorch_, _Keras_, _Tensorflow_ or _NNabla_, the technical challenge of developing a music separation system appears to be very low at first glance.
However, the lack of domain knowledge about the specifics of music signals often results in poor performance where issues are difficult to track using learning-based algorithms.
We therefore designed _Open-Unmix_ to address these issues by relying on procedures that were verified by the community or have proven to work well in the literature.

## Design Choices

The design choices made for _Open-Unmix_ have sought to reach two somewhat contradictory objectives. Its first aim is to have state-of-the-art performance, and its second aim is to still be easily understandable, so that it can serve as a basis for research to allow improved performance in the future. In the past, many researchers faced difficulties in pre- and post-processing that could be avoided by sharing domain knowledge.
Our aim was thus to design a system that allows researchers to focus on A) new representations and B) new architectures.

### Framework specific vs. framework agnostic

We choose _PyTorch_ to serve as a reference implementation due to its balance between simplicity and modularity [@openunmixpytorch]. Furthermore, we already ported the core model to [NNabla](https://github.com/sigsep/open-unmix-nnabla) and plan to release a port for Tensorflow 2.0, once the framework is released. Note that the ports will not include pre-trained models as we cannot make sure the ports would yield identical results, thus leaving a single baseline model for researchers to compare with.

### "MNIST-like"

Keeping in mind that the learning curve can be quite steep in audio processing, we did our best for _Open-unmix_ to be:

- __simple to extend__: The pre/post-processing, data-loading, training and models part of the code is isolated and easy to replace/update. In particular, a specific effort was done to make it easy to replace the model.
- __not a package__: The software is composed of largely independent and self-containing parts, keeping it easy to use and easy to change. 
- __hackable (MNIST like)__: Due to our objective of making it easier for machine-learning experts to try out music separation, we did our best to stick to the philosophy of baseline implementations for this community. In particular, _Open-unmix_ mimics the famous MNIST example, including the ability to instantly start training on a dataset that is automatically downloaded.

### Reproducible

Releasing _Open-Unmix_ is first and foremost an attempt to provide a reliable implementation sticking to established programming practice as were also proposed in [@mcfee2018open]. In particular:

- __reproducible code__: everything is provided to exactly reproduce our experiments and display our results.
- __pre-trained models__: we provide pre-trained weights that allow a user to use the model right away or fine-tune it on user-provided data [@umx; @umxhq].
- __tests__: the release includes unit and regression tests, useful to organize future open collaboration using pull requests.

## Results

![Boxplots of evaluation results of the `UMX` model compared with other methods from [@sisec18] (methods that did not only use MUSDB18 for training were omitted)\label{boxplot}](boxplot.pdf)

_Open-Unmix_ is based on the bi-directional LSTM model from [@uhlich17] and we compared it to other separation models that were submitted to the last SiSEC contest [@sisec18]. The results of `UMX` are depicted in \ref{boxplot}. It can be seen that our proposed model reaches state-of-the-art results. There is no statistically significant difference between the best method `TAK1` and `UMX`. Because `TAK1` is not released as open-source, this indicates that _Open-Unmix_ is the current state-of-the-art open-source source separation system.

# Community

Open-Unmix was developed by Fabian-Robert Stöter and Antoine Liutkus at Inria Montpellier.
The research concerning the deep neural network architecture as well as the training process was done in close collaboration with Stefan Uhlich and Yuki Mitsufuji from Sony Corporation.

In the future, we hope the software will be well received by the community. _Open-Unmix_ is part of an ecosystem of software, datasets, and online resources: the __sigsep__ community.

First, we provide MUSDB18 [@rafii17] and MUSDB18-HQ [@musdb18hq] which are the largest freely available datasets; this comes with a complete toolchain to easily parse and read the datasets [@musdb].
We maintain _museval_, the most used evaluation package for source separation [@museval].
We also are the organizers of the largest source separation evaluation campaign such as [@sisec18]. In addition, we implemented a reference implementation using a multi-channel Wiener filter, released in [@norbert]. The `sigsep` community is organized and presented on its [own website](https://sigsep.github.com). _Open-Unmix_ itself can be found on [https://open.unmix.app](https://open.unmix.app), which links to all other relevant sites and provides further information, such as audio demos.

## Outlook

_Open-Unmix_ is a community-focused project. We therefore encourage the community to submit bug-fixes and comments and improve the computational performance. However, we are not looking for changes that only focus on improving the separation performance as this would be out of scope for a baseline implementation. Instead, we expect many researchers will fork the software as a basis for their research and the documentation explicates several custom options to extend the code (shown [here](https://github.com/sigsep/open-unmix-pytorch/blob/master/docs/extensions.md)).

# References
