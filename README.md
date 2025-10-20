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

1. Quality control, normalization, and integration of single-cell transcriptomic and proteomic data.

2. Malignant vs. Normal Cell Identification

3. KNN smoothing for robust classification of malignant and healthy cells.

4. Clustering and State Identification

5. K-means clustering based on CNMF-derived gene programs.

6. Identification of gene modules correlated with LSC and blast states.

7. Target Identification

8. Analysis of surface ADT markers.

9. Combinatorial testing to identify candidate immunotherapy targets.

10. Validation using bulk RNA-seq and independent single-cell datasets.
