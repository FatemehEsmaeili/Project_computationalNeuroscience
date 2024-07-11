# Project-Computational-Neuroscience
This repository is about project of computational neuroscience 

## 1.Problem statement
  Accurately decoding motor parameters from brain signals is a fundamental challenge in neuroscience, with significant implications for brain-computer interface (BCI) development and neuroprosthetics. Traditional methods often struggle with the complexity and high dimensionality of electrophysiological data, which can lead to overfitting and poor generalization. The goal of this study is to develop an effective method for decoding hand and arm movements from electrocorticography (ECoG) signals recorded from macaque monkeys. This involves preprocessing the ECoG data, transforming it to capture relevant time-frequency information, and applying robust regression techniques to accurately predict motor parameters. By addressing these challenges, we aim to enhance the performance and reliability of BCIs, ultimately improving the quality of life for individuals with motor impairments.

## 2.Related works
  Chao ZC, Nagasaka Y, Fujii N. Long-term asynchronous decoding of arm motion using electrocorticographic signals in monkey. Frontiers in neuroengineering. 2010 Mar 30;3:1189

## 3.Proposed Method
Our proposed method involves a comprehensive approach to decode motor parameters from ECoG signals using advanced preprocessing and regression techniques. The methodology can be divided into several key steps: signal preprocessing, wavelet transformation, and partial least squares (PLS) regression for decoding

## 4.Implementation

### 4.1. Dataset
Dataset was collected from http://www.www.neurotycho.org/food-tracking-task.
The dataset includes electrophysiological and behavioral recordings from two Japanese macaques (A and K) with chronically implanted multichannel ECoG electrode arrays in the subdural space. For monkey A, 13 experiments were conducted over a 3-month period using a Neuralynx Digital Lynx data acquisition system, followed by 10 experiments over the next 5 months using a Neuralynx Digital Falcon telemetry system. For monkey K, 12 experiments were performed over 2 months using a Cyberkinetics system. ECoG signals were recorded at a sampling rate of 1 kHz per channel, and the monkeys' movements were captured at 120 Hz using a Vicon optical motion capture system.
Each monkey wore a custom-made jacket with reflective markers on the shoulders, elbows, and wrists, and their heads were restrained with a custom-made helmet. In some experiments, five additional markers were placed on the reaching hand to determine arm joint angles.

### 4.2. Model
In our processing methodology, ECoG signals recorded at 1 kHz were band-pass filtered from 0.1 to 600 Hz and re-referenced using a common average reference. 
We generated time-frequency representations of ECoG signals using Morlet wavelet transformations at ten center frequencies, forming 10 × 10 scalogram matrices resampled at ten time lags. These matrices were then normalized to z-scores. To predict motor parameters, we used partial least squares (PLS) regression, pooling normalized scalogram matrices from all electrodes into high-dimensional predictor vectors. This allowed us to model motor parameters as linear combinations of these predictor vectors, with weights estimated through PLS regression to avoid over-fitting. The optimal number of PLS components was determined via 10-fold cross-validation, and R² values were calculated to assess the explained variance.

![image](https://github.com/FatemehEsmaeili/Project_computationalNeuroscience/assets/59010636/54464c4e-aabd-40fa-bf8c-1060139ddbf8)

### 4.3. Evaluate
By focusing on 𝑅,R2 NRMS metrics, we aimed to assess both the strength of correlation and the accuracy of predictions in our model for decoding motor parameters from ECoG signals.

![image](https://github.com/FatemehEsmaeili/Project_computationalNeuroscience/assets/59010636/58f60b97-ebf0-4f67-8bb4-b08ea8e80041)



