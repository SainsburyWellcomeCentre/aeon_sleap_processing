# AEON SLEAP Processing

Project Aeon computational pose and ID tracking pipeline using SLEAP.

## Overview

This repository contains the code and models for the AEON project's pose and identity tracking pipeline. The pipeline is designed to track mice in the AEON arena, combining data from multiple camera views to generate high-quality tracking data with accurate identity assignments.

## Pipeline Structure

The pipeline consists of three main components:

1. [**Pose Model**](pose_model/README.md): Pre-trained SLEAP models for tracking body parts of mice from the top camera view

2. [**ID Model**](id_model/README.md): Process for training identity models using the zoomed-in quadrant cameras and mapping identities back to the top view

3. [**Pose and ID combining**](pose_id_combine/README.md): Tools for generating and combining pose and ID predictions

## Citation Policy

If you use this software, please cite it as below:

Sainsbury Wellcome Centre Foraging Behaviour Working Group. (2023). Aeon: An open-source platform to study the neural basis of ethological behaviours over naturalistic timescales,  https://doi.org/10.5281/zenodo.8411157

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8411157.svg)](https://zenodo.org/doi/10.5281/zenodo.8411157)