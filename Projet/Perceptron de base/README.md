## Introduction

À partir du livre Neural Networks and Deep Learning book de Michel Nielsen,
- mettre en oeuvre un perceptron « de base » (une couche de poids) 
- pour la reconnaissance de chiffres manuscrit à partir de MNIST dataset, en python et afficher les resultats obtenus
- tracer la courbe d'erreur pendant l'entraînement, qui montre comment l'erreur du modèle diminue au fil des itérations de l'entraînement
- afficher la matrice de confusion, qui montre combien de fois chaque classe a été prédite correctement ou incorrectement.

## Mise en oeuvre

1. Extraction des données à partir des fichiers .gz en utilisant la bibliothèque gzip de Python

2. Définition du perceptron "de base". Dans ce cas, nous utiliserons une couche de poids et la fonction d'activation sigmoïde. Nous initialiserons les poids aléatoirement

3. Entraînement du perceptron en utilisant l'algorithme de rétropropagation pour ajuster les poids en fonction des erreurs. Vous pouvez utiliser la fonction de coût de l'entropie croisée pour mesurer l'erreur. Dans ce cas, nous utiliserons un taux d'apprentissage de 0,1 et un lot d'entraînement de 10 images.
> On trace la courbe d'erreur pendant l'entraînement, qui montre comment l'erreur du modèle diminue au fil des itérations de l'entraînement

4. Évaluation de la précision du modèle en utilisant les données de test. Cela affiche l'exactitude (accuracy) du modèle, qui est la proportion de prédictions correctes sur l'ensemble de données de test.
> On affiche la matrice de confusion, qui montre combien de fois chaque classe a été prédite correctement ou incorrectement. La diagonale principale de la matrice représente le nombre de prédictions correctes pour chaque classe

## Conclusion 

Ce modèle de perceptron « de base » ne donne pas une précision très élevée sur le MNIST dataset, car il ne peut modéliser que des relations linéaires entre les pixels des images. Des réseaux de neurones plus avancés avec des couches cachées et des fonctions d'activation non linéaires sont nécessaires pour atteindre une précision élevée sur ce type de tâche.