# AEON SLEAP Processing

Project Aeon computational pose and ID tracking pipeline using SLEAP.

## Overview

This repository contains the code and models for the Aeon project's pose and identity tracking pipeline. The pipeline is designed to track mice in the Aeon arena, combining data from multiple camera views to generate high-quality tracking data with accurate identity assignments.

## Pipeline Structure

The pipeline consists of three main components:

1. [**Pose Model**](pose_model/README.md): Pre-trained SLEAP models for tracking body parts of mice from the top camera view

2. [**ID Model**](id_model/README.md): Process for training identity models using the zoomed-in quadrant cameras and mapping identities back to the top view

3. [**Pose and ID combining**](pose_id_combine/README.md): Tools for generating and combining pose and ID predictions

## Analysis Tools

[**Social Behavior Detection**](social_behavior_detection/README.md): Automated detection of social behaviors (fights, tube tests) from the pipeline's pose and identity tracking outputs

## Citation Policy

If you use this software, please cite it as below:

D. Campagner, J. Bhagat, G. Lopes, L. Calcaterra, A. G. Pouget, A. Almeida, T. T. Nguyen, C. H. Lo, T. Ryan, B. Cruz, F. J. Carvalho, Z. Li, A. Erskine, J. Rapela, O. Folsz, M. Marin, J. Ahn, S. Nierwetberg, S. C. Lenzi, J. D. S. Reggiani, SGEN group – SWC GCNU Experimental Neuroethology Group. _Aeon: an open-source platform to study the neural basis of ethological behaviours over naturalistic timescales._ Preprint at https://doi.org/10.1101/2025.07.31.664513 (2025)

[![DOI:10.1101/2025.07.31.664513](https://img.shields.io/badge/DOI-10.1101%2F2025.07.31.664513-AE363B.svg)](https://doi.org/10.1101/2025.07.31.664513)