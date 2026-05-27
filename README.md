# Penguin Species Classifier

## Description

This project is a web application that classifies penguin species based on their physical measurements. It uses the Palmer Penguins dataset and a machine learning model trained with scikit-learn.

The user can enter measurements such as bill length, bill depth, flipper length, body mass, island, and sex. The application then predicts the most likely penguin species and displays the prediction probabilities.

The project also includes a KMeans clustering visualization to show how penguins can be grouped based on their physical characteristics.

## Features

- Web interface built with Flask
- Machine learning model trained with scikit-learn
- Penguin species prediction from user input
- Prediction probabilities for each species
- KMeans clustering visualization
- Responsive HTML and CSS design

## Dataset

The project uses the Palmer Penguins dataset. The dataset contains measurements from three penguin species:

- Adelie
- Chinstrap
- Gentoo

The features used by the model are:

- Bill length
- Bill depth
- Flipper length
- Body mass
- Island
- Sex

## Machine Learning

The main classifier is a Random Forest model. The model is trained using numerical and categorical features. Categorical values such as island and sex are processed using one-hot encoding.

The project also uses KMeans clustering as an unsupervised learning visualization. KMeans does not use the species labels during clustering. Instead, it groups penguins based only on their measurements.

## Technologies Used

- Python
- Flask
- pandas
- scikit-learn
- seaborn
- matplotlib
- HTML
- CSS
- uv

## Project Structure

```text
penguin-classifier/
├── app.py
├── train_model.py
├── generate_kmeans_plot.py
├── data/
│   └── penguins.csv
├── models/
│   └── penguin_model.pkl
├── static/
│   ├── styles.css
│   └── images/
│       └── species_clusters.png
├── templates/
│   ├── layout.html
│   ├── index.html
│   └── predict.html
├── README.md
├── pyproject.toml
└── uv.lock