Project Overview

This repository contains the code and analysis associated with our study on pediatric acute myeloid leukemia (pAML). The study focuses on understanding the genetic and phenotypic heterogeneity of pAML using single-cell transcriptomic and surface proteomic (CITE-seq) profiles.

The dataset includes 27 samples from 16 high-risk pAML patients and 5 healthy donors, enabling detailed characterization of malignant and healthy hematopoietic cell populations. Our goal is to support the development of targeted immunotherapies for pAML.

Dataset Description

Type: Single-cell RNA-seq and CITE-seq

Samples: 27 total

16 high-risk pAML patients

5 healthy donors

Source: Pediatric bone marrow samples

Analysis Pipeline

The repository contains code implementing the following analyses:

Data Preprocessing

Quality control, normalization, and integration of single-cell transcriptomic and proteomic data.

Malignant vs. Normal Cell Identification

KNN smoothing for robust classification of malignant and healthy cells.

Clustering and State Identification

K-means clustering based on CNMF-derived gene programs.

Identification of gene modules correlated with LSC and blast states.

Target Identification

Analysis of surface ADT markers.

Combinatorial testing to identify candidate immunotherapy targets.

Validation

Comparison and validation using bulk RNA-seq and independent single-cell datasets.
