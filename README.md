# MADI Project : Implementation of prudent classifier, "Classification by pairwise coupling of imprecise probabilities"

This repository contains the code for the project of the course MADI (Modèles et Algorithmes pour la Décision dans l'Incertain).

## Team members

- BOUTON Jules
- SUN Amélie

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [Todo](#todo)
- [References](#references)

## Introduction

We choose to implement the prudent classifier presented in the article [1]. It consists of a pairwise coupling of imprecise probabilities. The idea is to decompose the multi-class classification problem into a set of binary classification problems. The imprecise classifier is then built by combining the results of the binary imprecise classifiers. In our case, we choose to use decision trees as binary classifiers as it was the choice made in the article. We will also test the different decomposition schemes proposed in the article.

## Requirements

- Python 3.x
- NumPy library
- SciPy library
- sklearn library
- ucimlrepo library
- pandas library
- matplotlib library

You can install the libraries using pip:

```bash
pip install numpy
pip install scipy
pip install sklearn
pip install ucimlrepo
pip install pandas
pip install matplotlib
```

## Usage

To rerun the experiments, you can simply execute the experiment_digits.py file :

```bash
python source/experiment_digits.py
```

The seed is fixed at the beginning of the file to ensure reproducibility of the results (see comment to pick the right seed). Instructions are given in the file to choose the dataset and the decomposition scheme. The results are saved in the out folder.

## Todo

- [x] Génération de jeux de données synthétiques représentable graphiquement;

- [x] Implémentation de la classification prudente de l'article [1] :
  
  - [x] 4 schémas de décomposition (OVO, OVA, ECOC dense, ECOC sparse);
  
  - [x] Imprécision sur les prédictions (Bayesian credibility interval bounds obtained using a beta prior with parameters aj = bj = 3.5);

  - [x] Correction des prédictions erronées pour la phase de test (Unconditional discounting);

  - [x] Règle de la maximalité pour combiner (Maximality rule).

- [x] Expérimentations sur les jeux de données synthétiques;

- [x] Expérimentations sur les jeux de données réels;

- [x] Rapport.

## References

- [1] Benjamin Quost, Sébastien Destercke. Classification by pairwise coupling of imprecise probabilities. Pattern Recognition, 2018, 77, pp.412-425. ⟨10.1016/j.patcog.2017.10.019⟩. ⟨hal-01652798⟩
