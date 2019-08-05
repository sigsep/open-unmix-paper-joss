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

Music source separation is the task of decomposing music into its constitutive components, e.g., yielding separated stems for the vocals, bass and drums.
Such a separation has many applications ranging from rearranging/repurposing the stems (remixing, repanning, upmixing) to full extraction (karaoke, sample creation, audio restoration).
Music separation has a long history of scientific activity as it is known to be a very challenging problem.
In recent years, deep learning based systems - for the first time - yielded high quality separations that also lead to increased commercial interest.
However, until now, no open-source implementation that achieves state-of-the-art results was available.
_Open-Unmix_ closes this gap by providing a reference implementation based on deep neural networks.
It serves two main purposes: Firstly, to accelerate academic research as _Open-Unmix_ provides implementations for the most popular deep learning frameworks, giving researchers a flexible way to reproduce results; Secondly, we provide a pre-trained model for end users and even artists to try and use source separation.
Furthermore, we designed _Open-Unmix_ to be one core component in an open ecosystem on music separation, where we already provide open datasets, software utilities and open evaluation to foster reproducible research as the basis of future development.

# Background

Music separation is a problem which fascinated researchers since over 50 years. This is partly due to the fact that, mathematically, there exists no closed-form solution when many sources (instruments) are captured by a single microphone.
As the problem is impossible to solve without some assumptions when there are less mixtures than sources, researchers exploited further knowledge about the way the signals were recorded and mixed to make separation possible. A large number of these methods are centered around "classical" signal processing methods that revolve around assumptions concerning the sources to separate. For a more detailed overview see [@rafii17] and [@cano19].
<!-- - Explain how and when deep neural network based music separation outperformed existing methods. -->
Many of these classical signal processing based methods were hand-crafted and tuned to a small number of music recordings [@sisec10; @sisec11; @sisec13].
Systematic objective evaluation of these methods, however, was hardly feasible as freely available datasets did not exist at that time.
In fact, for a meaningful evaluation, the ground truth separated stems are necessary.
However, these are considered precious assets of the music mixing and mastering and, therefore, are usually not available freely.
This is because any commercial music is well known to be subject to copyright protection laws and, hence, is not generally available for re-distribution.
Nonetheless, in the past five years, freely available datasets were released that enabled the development of data-driven methods thanks to some artists that choose compatible licenses like Creative Commons.
Since then, progress in performance was closely linked to the availability of more data that allowed the use machine learning based methods.
This lead to a large performance boost similar to other audio tasks such as automatic speech recognition (ASR) where large amount of data was available.
In fact, in 2016 the speech recognition community had access to datasets with more than 10.000 hours of speech [@amodei16].
At the same time, the _MUSDB_ dataset was released [@rafii17] which comprises 150 full length music tracks – a total of just 10 hours of music.
To date, this is still the largest freely available dataset for source separation.
Nonetheless, even with this small amount of data, deep neural networks (DNNs) were not only successfully used for music separation but they are now setting the state-of-the art in this domain as can be seen by the results of the community-based signal separation evaluation campaign (SiSEC) [@sisec15; @sisec16; @sisec18].

In these challenges, the proposed systems are compared to other methods as well as a number of oracle methods [cite_oracles] to indicate possible upper bounds. On the other end, some of the classical signal processing based methods, known to be _reliable_, _fast_, _simple to use_, were treated as reference algorithms.
In fact, in the past, a number of systems and libraries were explicitly designed with these aspects in mind.
While today there exist also a number of commercial systems such as _Audionamix XTRAX STEMS_, _IZOTOPE RX 7_ or _AudioSourceRE_ which target end-users, we considered only tools that are available as open source software, suitable for research.

<!-- list of methods -->
The first publicly available software for source separation was _openBlissart_, released in 2011 [@weninger11]. It is written in C++ and represents the class of systems that are based on non-negative matrix factorization (NMF). In 2012, the _Flexible Audio Source Separation Toolbox (FASST)_ was presented in [@ozerov2011general; @salaun14]. It is written in MATLAB/C++ and is also based on NMF methods but additionally also included other model-based methods.
In 2016, the _untwist_ library was proposed in [@roma16]. It comprises a number of methods, ranging from classical signal processing based methods to feed forward neural networks. The library is written in Python 2.7. Unfortunately, it was not updated since 2 years and many methods are not tested.
_Nussl_ is a very recent framework, presented in [@manilow18]. It includes a large number of methods and generally focuses on classical signal processing methods rather than machine-learning based techniques. It has built-in interfaces for common evaluation metrics and data sets. The library offers great modularity and a good level of abstraction. However, it also means that it is challenging for beginners who might only want to focus on changing the machine learning parts of the techniques.

<!-- ## The gap -->
The main problem with all of these available implementations is that they do not deliver state-of-the-art results. In fact, no open-source system exists today that matches the performance of the best system proposed more than 4 years ago by [@uhlich15].
We believe that the lack of such a baseline has a serious negative impact on future research on source separation.
In fact, many new methods that were published in the last years, are usually compared to their own baselines, thus showing relative instead of absolute performance gains, so that other researchers cannot assess if a method performs as good as state-of-the-art.
Also, the lack of a common reference for the community potentially miss-guides young researchers and students that enter the field of music separation. The result of this can be observed by looking at the popularity of the above mentioned music separation frameworks on GitHub: all of the frameworks mentioned above combined are less popular than two recent deep learning papers that were accompanied by code (nussl: 196 stars, untwist: 95, pyfasst 79, openblissart 77) such as `MTG/DeepConvSep` from [@chandna17] (340 stars) and `f90/Wave-U-Net` from [stoller] (300 stars). Thus, users might be confused regarding which of these implementations can be considered state-of-the-art.

# Open-Unmix

Today, many new research in signal processing comes from applying machine learning to specific tasks such as music separation.
With the rise of simple to use machine learning frameworks such as [Keras], [Tensorflow], [Pytorch] or [NNabla], the technical challenge of developing a music separation system appears to be very low at first sight.
However, the lack of domain knowledge about the specifics of music signals often results in weak performance where issues are difficult to track using learning based algorithms.
We could observe this problem in [@sisec16; @sisec18], when we organized the source separation evaluation campaign. Computationally complex deep models might underperform simpler models just because of subtle differences in pre- and postprocessing. These problems could obviously have been discussed in a more systematic manner if the proposed methods would have been open-source.
We therefore designed _Open-Unmix_ to address these issues by relying on procedures that were verified by the community or proven to be working well by literature.

## Design Choices

The design choices made for _Open-unmix_ have sought to reach two somewhat contradictory objectives. Its first aim is to have state-of-the art performance, and its second aim is to still be easily understandable, so that it could serve as a basis for research allowing improved performance in the future. In the past, many researchers faced difficulties in pre- and post-processing that could be avoided by with sharing domain knowledge.
The aim was hence to design a system that allows researchers to focus on:

- new representations, in which case there is a need for established separation architectures to use as a basis for evaluation.
- new architectures, which needs solid pre and post-processing pipelines.

In short, _Open-Unmix_ inputs mixtures in the waveform domain, transforms them using a time-frequency representation, before applying a series of non-linear layers to predict the spectrogram of a target source. At the end of the chain, a postprocessing system gathers the estimates for all sources, and combines them through a multichannel Wiener filter to obtain the actual separated waveforms.

The most critical aspects of the choices can be summarized by stating that the system heavily relies on __expert-knowledge__: while end-to-end systems that directly produce estimates in the waveform (time) domain are a promising research direction, they currently do not lead to state-of-the-art performance in music separation.
On the contrary, systems operating in the time-frequency (TF) domain are still observed to significantly outperform more "modern" solutions that would bypass the expert knowledge required by TF processing.

From the perspective of _Open unmix_, signal processing fundamentals are encapsulated in _pre_ and _post-processing_ operations. _Preprocessing_: the computation of Short-Term Fourier Transforms (STFT). _Postprocessing_: the spectrogram of the estimates is obtained by multiplying element-wise the input spectrogram through a _mask_ whose values lie between 0 and 1. This comes from the knowledge that energies of the sources roughly __add up__ to the energy of the mixture. Then, multichannel Wiener filters exploit the spectrogram estimates to produce the actual waveforms of the estimates. STFT forward and inverse transformations are used as implemented in standard libraries. Wiener filtering is used as implemented in the [`sigsep.norbert`](https://github.com/sigsep/norbert) repository.

### Framework specific vs. framework agnostic

_Open-unmix_ is developed in parallel for multiple frameworks to cover the most number of users. Currently we support Tensorflow/Keras, PyTorch, NNabla, which together can be considered a sufficient coverage for our purpose. More specifically, the _PyTorch_ version will serve as the reference version due to its simplicity and modularity.
Likewise, the NNabla is close to the PyTorch code.
The tensorflow version will be release later when Tensorflow 2.0 is stable and will be more production-oriented (inference).

### Hackable, Fast and Simple

Keeping in mind that the leaarning curve can be quite steep in audio processing, we did our best for _Open-unmix_ to be:

- __simple to extend__: the pre/post processing, data-loading, training and models part of the code are clearly isolated and easy to replace/update. In particular, an extra-effort was done on making it easy to replace the core deep model that is currently a baseline BLSTM.
- __not a package__: keeping it easy to use and change the code made us design the software to be composed of largely independent and self-containing parts.
- __hackable (MNIST like)__: due to our objective of making it easier for machine-learning experts to try out music separation, we did our best to stick to the philosophy of baseline implementations for this community. In particular, _Open-unmix_ mimicks the famous MNIST example including the ability to instantly start training on data that is automatically downloaded.

### Reproducible

Releasing _Open-Unmix_ is first and foremost an attempt to provide a reliable implementation sticking to established programming practice. In particular:

- __reproducible code__: everything is provided to exactly reproduce our experiments and display our results.
- __pre-trained models__: we provide pre-trained weights that allow to use the model right away or fine-tune it on user-provided data.
- __tests__: the release includes unit tests and regression tests usefule to organize future open collaboration using pull-requests.

## Technical Details

![Block diagram of _Open-Unmix_\label{block_diagram}](UMX_BlockDiagram.pdf)

We will give now more technical details about _Open-Unmix_. Fig. \ref{block_diagram} shows the basic approach. During training, we learn a DNN which can be later used for separating songs.

### Datasets and Dataloaders

When designing a machine-learnig based method, our first step was to encapsulate cleanly the data-processing aspects.

- __Datasets__: we support the [MUSDB18] dataset, that is most established dataset that we released some years ago. Several other custom options are supported through native dataset and dataloader APIs. This encourages the interested researcher to train _Open-unmix_ model with her/his own data.
- __Efficient dataloading and transforms__: since preparing the batches for training is often the efficiency bottleneck, extra-care was taken to optimize speed and performance. Here, we use framwork specific data loading API instead of a generic module. For all frameworks we use the builtin STFT transform operator that can work on the GPU (Cite paper from Choi NIPS).
- __Essential augmentations__: data augmentation techniques for source separation are described in  [@uhlich17; something_from_icassp19] which we adopted here. They enable to attain good performance even though the audio datasets such as MUSDB18 are often of limited size.
- __Post processing__: add details to norbert. <!-- TODO: Antoine -->

### Model

![Separation network\label{separation_network}](https://docs.google.com/drawings/d/e/2PACX-1vTPoQiPwmdfET4pZhue1RvG7oEUJz7eUeQvCu6vzYeKRwHl6by4RRTnphImSKM0k5KXw9rZ1iIFnpGW/pub?w=959&h=308)

Fig. \ref{separation_network} gives the details about the used DNN model.

![General processing pipeline\label{processing_pipeline}](General_Processing_Pipeline.pdf)

<!-- TODO: Stefan -->
<!-- Add note about phase (we are only focusing on magnitudes for separation). -->

The system is trained to predict a separated source from the observation of its mixture with other sources. The corresponding training is done in a _discriminative_ way, i.e. through a dataset of mixtures paired with their true separated sources. These are used as ground truth targets from which gradients are computed. Although alternative ways to train a separation system have emerged recently, notably through _generative_ strategies trained through adversarial cost functions, they still did not lead to comparable performance.
Even if we acknowledge that such an approach could in theory allow to scale the size of training data since it can be done in an _unpaired_ manner, we feel that this direction is still in progress and cannot be considered state-of-the-art today.
That said, the _Open-Unmix_ system can easily be extended to such generative training, and the community is much welcome to exploit it for that purpose.

The constitutive parts of the actual deep model used in _Open-Unmix_ only comprise very classical elements.
Among them, we can mention:

  - _LSTM_ [@Hochreiter97] <!-- TODO: Stefan -->
  - _Fully connected time-distributed layers_ are used for dimension reduction and augmentation, at the input and output sides, respectively. They allow control over the number of parameters of the model and prove to be crucial for generalization.
  - _Skip connections_ are used in two ways: i/ the output to recurrent layers are augmented with their input, and this proved to help convergence. ii/ The output spectrogram is computed as an element-wise multiplication of the input. This means that the system actually has to learn _how much each TF bin does belong to the target source_ and not the _actual_ value of that bin. This is _critical_ for obtaining good performance and combining the estimates given for several targets, as done in _Open-unmix_.
  - _Non linearities_ are of three kinds: i/ rectified linear units (ReLU) allow intermediate layers to comprise nonnegative activations, which long proved effective in TF modelling. ii/ `tanh` are known to be necessary for a good training of LSTM model, notably because they avoid exploding input and output. iii/ a `sigmoid` activation is chosen before masking, to mimic the way legacy systems take the outputs as a _filtering_ of the input.
  - _Batch normalization_ long proved important for stable training, because it makes the different batches more similar in terms of distributions. In the case of audio where signal dynamics can be very important, this is crucial.

### Training

Our experience gained during the research we did for releasing _Open-Unmix_ taught us that successful __training__ of the model takes expert knowledge that we want to share with the community, since only an implementation can enable widespread diffusion of these techniques.
Indeed, those tricks are often deemed of not sufficient scientific importance to be found in scientific papers.

In particular, we use the following setup for training.
During training, we learn the weights of the BLSTM by minimizing the mean squared error (MSE) with the ADAM optimizer [@kingma2014adam].
Data augmenation is an important step due to the small size of MUSDB and, therefore, we use the data augmentation as described in [@uhlich17].
<!-- - Cost functions (supervised/adversarial training, clustering-based) -->
<!-- TODO: Stefan -->
<!-- - Optimization algorithms, learnig rate, scheduling, early stopping, etc.-->
<!-- - Data augmentations.-->
<!-- - Model hyperparameters.-->
<!-- - Temporal context, batch size, standardization/scaling, regularization-->
<!--
## Usage and Results

### Objective Evaluation

compare and list link to demo website

### For Artists

torchhub
-->

# Contributions

## Development

Open-Unmix was developed by Fabian-Robert Stöter and Antoine Liutkus at Inria Montpellier.
The research concerning the deep neural network architecture as well as the training process was done in close collaboration with Stefan Uhlich and Yuki Mitsufuji from Sony Corporation.

## Community

In the future, we hope the software will be well received by the community. As stated in our contributors agreement.

## The source separation community

- Open-unmix is part of a ecosystem of software, datasets and online resources: the `sigsep` community
- we provide MUSDB18 and MUSDB18-HQ, the largest freely available dataset, this comes with a complete toolchain to easily parse and read the dataset such as [musdb] and [mus]
- [museval], is mostly used evaluation package for source separation
- we also are the organizers of the largest source separation evaluation campaign
- in this campaign we noticed: previous state-of-the-art systems could not be matched by newer systems (e.g. UHL2)

# References
