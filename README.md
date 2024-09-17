# Flax-Based Denoising Autoencoder for NMR CPMG T2 Relaxometry Data

This repository contains a Flax-based Denoising Autoencoder, designed as an initial test to remove low-frequency noise from noisy multi-exponential decay data. The data originates from NMR CPMG T2 relaxometry signals, which are used to assess fluid levels in patients with chronic kidney disease (CKD).

## Project Overview

In my current research at the **Cima Lab** in the **MIT Koch Institute**, we aim to improve the processing of NMR data for more accurate fluid level identification in CKD patients. This specific project focuses on using a Denoising Autoencoder (DAE) to clean noisy exponential decay signals, allowing for more accurate analysis and diagnosis.

This repository implements a modified version of the **Variational Autoencoder** example found in the [Flax documentation](https://github.com/google/flax/tree/main/examples/vae).

## Features
- **Denoising Autoencoder**: Trained to remove low-frequency noise from multi-exponential decay data.
- **Flax-based Implementation**: Leverages the powerful Flax library for neural networks in JAX, optimized for machine learning research.
- **Noise Reduction**: The model is specifically tailored to handle noisy NMR CPMG T2 relaxometry signals.

## Data and Use Case

The data used in this project consists of noisy exponential decays from NMR signals, with the goal of denoising these signals to improve the detection of relaxation times, critical for monitoring fluid levels in CKD patients.

## References
- **Flax VAE example**: [https://github.com/google/flax/tree/main/examples/vae](https://github.com/google/flax/tree/main/examples/vae)
- **MIT Koch Institute** - [Cima Lab](https://cima-lab.mit.edu/)

## Future Work
This project serves as a foundational experiment. Future developments will explore the application of different architectures and fine-tuning methods to further improve noise removal and signal processing for these noisy tri-exponential decays.

## Adjacent Analyses

The research that I have conducted thus far on denoising multi-exponential decays includes but is not limited to:
- **Wavelet Denoising**: Applying wavelet transforms to reduce high-frequency noise while preserving signal integrity.
- **Analytical Solving Methods**: including the Pade-Laplace method and the Prony method.
- **Optimization Techniques**: methods such as non-linear least squares regression and Bayesian parameter estimation.
- **Multi-Exponential Decay Resolution Analysis**: Investigating the limits of resolving power for separating closely spaced decay constants. A good rule of thumb is that you wont resolve two peaks within 2x of one another with an SNR of less than 200.
- **Filtering Methods**: Including but not limited to the moving average, mean displacement ratio, and the Provencher filter.

The bible for multi-exponential analysis can be found [here](https://pubs.aip.org/aip/rsi/article/70/2/1233/438854/Exponential-analysis-in-physical-phenomena).
