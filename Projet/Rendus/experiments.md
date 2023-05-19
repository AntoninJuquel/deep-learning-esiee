# Introduction au Deep Learning et aux Réseaux de Neurones

Dans ce projet, nous allons construire des modèles de deep learning pour résoudre une tâche de reconnaissance d'images de chiffres manuscrits. Nous commencerons par un perceptron simple, puis nous ajouterons des couches pour créer un MLP, et enfin, nous expérimenterons avec un CNN. Nous évaluerons et comparerons les performances de ces différents modèles pour comprendre leurs forces et leurs faiblesses.


Dans ce projet nous utiliserons :
- Le jeu de données **MNIST**, qui est largement utilisé pour ce type de tâches. 
- **TensorFlow**, qui comprend Keras pour la construction de modèles de deep learning. 
- **Matplotlib** pour tracer les courbes d'apprentissage pour montrer l'évolution de la précision et de la perte sur les ensembles d'entraînement et de validation au fil des époques.
- **Seaborn** pour afficher la matrice de confusion pour nous montrer le nombre de fois où chaque classe a été prédite correctement ou incorrectement.

Pour commencer nous avons écrit une fonction pour créer des modèles de deep learning avec différentes architectures. Cette fonction prend en entrée une liste de couches et le nombre d'epochs et retourne un modèle Keras compilé avec les couches spécifiées. Nous utiliserons cette fonction pour créer des modèles avec différentes architectures et comparer leurs performances.

# 1. Classification des données MNIST par réseau multicouches

## Perceptron de base

Nous allons commencer par un perceptron simple sans couche cachée. Nous utiliserons la fonction `create_model` pour créer un modèle avec une couche d'entrée de 784 neurones (28x28 pixels) et une couche de sortie de 10 neurones (un pour chaque classe). Nous utiliserons la fonction d'activation la fonction d'activation `softmax` pour la couche de sortie. Nous utiliserons la fonction d'erreur `sparse_categorical_crossentropy` pour calculer la perte et la métrique `accuracy` pour calculer la précision.

Dans ces premières experience nous avons fait varier le nombre d'epochs.

- 5 epochs :
    ```
    Test accuracy: 0.9258000254631042
    Matrice de confusion: 
    [[ 963    0    2    2    0    4    6    2    1    0]
    [   0 1109    3    2    0    1    4    2   14    0]
    [   8    9  924   19    8    4   12   11   34    3]
    [   2    0   17  930    0   24    2   10   17    8]
    [   1    1    2    2  914    0   12    4   10   36]
    [  10    1    3   33    8  787   11    7   27    5]
    [   9    3    6    1    7   17  910    2    3    0]
    [   1    5   22    7    7    0    0  956    4   26]
    [  10    6    7   28    9   36    8   12  851    7]
    [  11    7    1   10   23   10    0   30    3  914]]
    ```
- 10 epochs :
    ```
    Test accuracy: 0.9290000200271606
    Matrice de confusion: 
    [[ 965    0    1    2    0    5    3    3    1    0]
    [   0 1115    4    2    0    1    3    2    8    0]
    [   6    8  925   19    9    3   10   10   39    3]
    [   2    0   17  929    1   20    2   11   21    7]
    [   1    1    5    3  922    0    5    5    9   31]
    [   8    2    1   38    9  772   13    9   34    6]
    [  13    3    9    2    8   14  906    1    2    0]
    [   1    6   23    3    5    1    0  954    2   33]
    [   7    7    7   22    9   19    5   11  879    8]
    [  10    8    1   11   26    4    0   21    5  923]]
    ```
- 30 epochs :
    ```
    Test accuracy: 0.9275000095367432
    Matrice de confusion: 
    [[ 956    0    0    2    1    7   10    3    1    0]
    [   0 1115    5    1    0    1    3    2    8    0]
    [   6    7  928   15    7    4   10   10   40    5]
    [   3    0   19  924    1   24    2    9   23    5]
    [   1    1    7    1  916    0   10    6    7   33]
    [   8    3    3   33    9  783   15    4   30    4]
    [  10    3    9    1    6   14  913    1    1    0]
    [   1    8   24    4    4    1    0  947    1   38]
    [   7    8    6   20    8   30   10    8  864   13]
    [  10    8    1    8   20    8    0   19    6  929]]
    ```

Les résultats des trois expériences montrent une amélioration de l'exactitude des tests avec l'augmentation du nombre d'époques, bien que l'augmentation ne soit pas linéaire.

Dans la première expérience, vous avez utilisé 5 époques et obtenu une précision de test de 92.58%. Dans la deuxième expérience, avec 15 époques, la précision a augmenté à 92.90%. Enfin, dans la troisième expérience, avec 30 époques, l'exactitude des tests a légèrement augmenté à 92.97%.

Cependant, il est important de noter que l'augmentation du nombre d'époques ne garantit pas toujours une amélioration de la précision du modèle. Au contraire, un trop grand nombre d'époques peut entraîner un surapprentissage, où le modèle s'adapte trop bien aux données d'entraînement et se généralise mal aux nouvelles données. Dans ces trois expériences, il semble que le modèle commence à montrer des signes de surapprentissage après environ 15 époques, car la précision de validation commence à se stabiliser voire à diminuer légèrement.

Les matrices de confusion montrent également que le modèle est généralement bon pour classer correctement les chiffres, avec la plupart des erreurs se produisant entre des chiffres qui ont des formes similaires.

En conclusion, selon ces résultats, il semble que le nombre optimal d'époques pour ce modèle spécifique soit d'environ 15. Au-delà de ce nombre, les gains en précision sont marginaux et le risque de surapprentissage augmente.

## Perceptron à deux couches

Nous passons maintenant à la création d'un perceptron à deux couches. Dans ce cas, nous allons ajouter une couche cachée à notre modèle. Cela permettra au réseau de neurones d'apprendre des représentations plus complexes des données. 

Pour ce faire nous avons ajouté une couche `Dense` avec `n` neurones et l'activation `ReLU` avant la couche de sortie. La fonction d'activation ReLU (Rectified Linear Unit) est couramment utilisée dans les réseaux de neurones profonds car elle permet d'apprendre des représentations non linéaires. 

Pour explorer l'effet du nombre de neurones dans la couche cachée sur les performances du modèle, nous avons répété ce processus avec différents nombres de neurones et comparé les résultats.

- 64 neurones dans la couche cachée:
    ```
    Test accuracy: 0.9721999764442444
    Matrice de confusion: 
    [[ 972    0    1    0    1    1    1    2    1    1]
    [   0 1125    2    1    0    2    2    1    2    0]
    [   3    2  999    3    2    0    2   13    8    0]
    [   0    0    7  979    0    4    0    8    2   10]
    [   0    0    6    1  959    0    2    4    1    9]
    [   2    0    0   24    3  849    7    1    3    3]
    [   4    4    3    1    9    1  935    1    0    0]
    [   0    4    7    0    2    0    0 1012    0    3]
    [   3    3    8    4    5    7    5    9  922    8]
    [   1    5    0    3   15    3    0   10    2  970]]
    ```

- 128 neurones dans la couche cachée:
    ```
    Test accuracy: 0.9768999814987183
    Matrice de confusion: 
    [[ 968    0    2    1    2    0    3    1    3    0]
    [   0 1121    4    1    0    2    2    2    3    0]
    [   1    2 1006    2    2    1    1    4   12    1]
    [   0    0    4  960    0   25    0    6    6    9]
    [   1    0    5    1  963    0    4    1    1    6]
    [   2    0    0    1    2  877    2    0    7    1]
    [   5    2    2    1    4   12  929    1    2    0]
    [   2    1    8    1    3    1    0 1005    3    4]
    [   2    0    1    1    4    6    0    3  953    4]
    [   1    2    0    0   10    5    0    2    2  987]]
    ```

- 256 neurones dans la couche cachée:
    ```
    Test accuracy: 0.98089998960495
    Matrice de confusion: 
    [[ 967    1    2    1    1    0    0    1    4    3]
    [   0 1121    4    0    0    2    3    2    3    0]
    [   4    0 1003    5    1    0    1    4   13    1]
    [   0    0    5  987    0    5    0    3    3    7]
    [   0    1    4    0  966    0    3    1    2    5]
    [   2    0    0    8    1  869    5    1    5    1]
    [   3    1    2    1    3    1  945    0    2    0]
    [   1    2    6    2    1    0    0 1009    3    4]
    [   3    0    2    4    4    2    0    2  955    2]
    [   2    2    0    1    8    1    0    6    2  987]]
    ```

- 512 neurones dans la couche cachée:
    ```
    Test accuracy: 0.982699990272522
    Matrice de confusion: 
    [[ 973    0    0    1    2    0    2    2    0    0]
    [   0 1123    3    1    0    0    1    4    3    0]
    [   6    0 1009    2    2    0    2    5    5    1]
    [   0    0    3  994    0    2    0    5    5    1]
    [   0    1    5    0  971    0    2    0    1    2]
    [   2    0    0    8    2  871    4    1    3    1]
    [   2    3    1    1    2    3  943    0    3    0]
    [   0    1    5    0    0    0    0 1018    3    1]
    [   5    0    1    4    5    0    1    6  948    4]
    [   2    2    0    4   10    5    0    7    2  977]]
    ```
Comme nous le constatons, l'ajout d'une couche cachée à notre perceptron a permis de produire des performances nettement améliorées, par rapport à une simple configuration de perceptron à une couche. C'est une démonstration de l'utilité des couches cachées pour apprendre des représentations plus complexes des données.

En comparant les performances des trois modèles, nous constatons que le nombre de neurones dans la couche cachée a un effet important. Augmenter le nombre de neurones dans la couche cachée de 64 à 128 a conduit à une amélioration de la précision du test. Toutefois, passer de 128 à 256 neurones n'a pas significativement amélioré la précision du test, ce qui pourrait indiquer que le modèle est peut-être déjà assez complexe avec 128 neurones pour représenter les données d'entraînement.

Notons que bien que l'augmentation du nombre de neurones puisse aider à capturer des représentations plus complexes, cela peut aussi conduire à un surapprentissage si le modèle devient trop complexe. Un modèle surappris peut avoir une performance exceptionnelle sur les données d'entraînement, mais il peut ne pas généraliser correctement à de nouvelles données. Ainsi, il est important d'équilibrer la complexité du modèle avec la quantité et la variété des données disponibles pour l'entraînement.

En outre, il est aussi important de noter que la matrice de confusion montre que les modèles ont tendance à faire certaines erreurs plus souvent que d'autres. Par exemple, certains chiffres peuvent être plus souvent confondus avec d'autres. Cela pourrait indiquer des limites de la capacité du modèle à distinguer certaines classes, ou cela pourrait refléter des caractéristiques particulières du jeu de données (par exemple, si certains chiffres sont écrits de manière très similaire).

## Perceptron multicouche

Nous allons créer un réseau de neurones profond en ajoutant plus de couches cachées à notre modèle. Comme avant, nous allons expérimenter avec différents nombres de couches et de neurones par couche, et observer l'effet sur les performances du modèle.

- 2 couches cachées à 128 neurones:
    ```
    Test accuracy: 0.9778000116348267
    Matrice de confusion: 
    [[ 970    0    1    1    1    2    1    0    2    2]
    [   0 1122    2    0    0    0    2    1    8    0]
    [   1    1 1013    0    1    0    2    9    5    0]
    [   0    0    6  968    0   16    0    7    9    4]
    [   0    0    4    0  961    0    7    2    2    6]
    [   2    0    0    5    1  870    4    1    7    2]
    [   4    1    0    1    2    4  946    0    0    0]
    [   1    0    8    0    0    0    1 1011    3    4]
    [   2    0    6    3    0    2    2    3  954    2]
    [   2    2    0    2   11    4    0   18    7  963]]
    ```

- 3 couches cachées à 128 neurones:
    ```
    Test accuracy: 0.9797000288963318
    Matrice de confusion: 
    [[ 968    1    0    1    0    4    2    1    2    1]
    [   0 1124    1    3    1    1    2    1    2    0]
    [   4    0 1010    5    1    0    0    4    7    1]
    [   1    0    1  996    0    1    0    3    6    2]
    [   0    0    3    1  956    0    5    1    2   14]
    [   2    0    0   15    0  868    1    0    5    1]
    [   3    2    1    0    2   10  936    0    4    0]
    [   0    2   12    5    1    0    0  993    5   10]
    [   0    0    2    4    0    4    0    0  962    2]
    [   0    1    1    4    6    4    0    4    5  984]]
    ```

- 4 couches cachées à 128 neurones:
    ```
    Test accuracy: 0.9786999821662903
    Matrice de confusion: 
    [[ 975    0    0    1    0    0    3    1    0    0]
    [   0 1124    0    3    0    0    2    3    3    0]
    [   3    0 1005    3    3    0    4   12    2    0]
    [   0    0    4  990    0    8    0    3    1    4]
    [   0    0    2    0  958    0    6    2    2   12]
    [   2    0    0    9    1  874    3    0    1    2]
    [   3    2    1    1    0    7  942    0    2    0]
    [   0    4    5    4    2    0    0 1002    1   10]
    [   5    3    8    9    0    6    0    5  934    4]
    [   2    2    0    5   10    4    0    3    0  983]]
    ```

- 5 couches cachées à 128 neurones:
    ```
    Test accuracy: 0.9810000061988831
    Matrice de confusion: 
    [[ 970    0    0    0    1    1    4    1    2    1]
    [   0 1129    2    3    0    0    0    1    0    0]
    [   4    1 1009    2    5    0    1    5    4    1]
    [   2    0    3  989    0    0    0    5    7    4]
    [   1    0    0    0  969    0    6    1    0    5]
    [   2    0    0   12    1  867    6    0    2    2]
    [   1    3    1    1    3    3  945    0    1    0]
    [   2    3   11    2    4    0    0  998    3    5]
    [   0    1    4    4    1    2    2    1  955    4]
    [   1    5    0    0   11    5    1    3    4  979]]
    ```

- Avec 2 couches cachées, le modèle atteint une précision de test de 0.9778 après 15 époques. Il semble qu'il y ait un certain surapprentissage, car la précision sur les données d'entraînement continue à augmenter, tandis que la perte sur les données de validation commence à augmenter après la 7ème époque. Cela indique que le modèle commence à mémoriser l'ensemble d'entraînement plutôt que d'apprendre des représentations généralisables.

- Avec 3 couches cachées, le modèle atteint une précision de test légèrement supérieure de 0.9797 après 15 époques. Tout comme le modèle à 2 couches cachées, ce modèle semble aussi présenter un surapprentissage, car la perte de validation commence à augmenter après la 7ème époque, alors que la précision sur l'ensemble d'entraînement continue à augmenter.

- Avec 4 couches cachées, le modèle atteint une précision de test de 0.9792, qui est légèrement inférieure à celle du modèle à 3 couches cachées mais supérieure à celle du modèle à 2 couches cachées. Encore une fois, il semble y avoir un surapprentissage, car la perte de validation commence à augmenter après la 6ème époque, alors que la précision sur l'ensemble d'entraînement continue à augmenter.

Globalement, l'ajout de couches cachées a permis d'améliorer légèrement la précision des tests, mais cela a également conduit à un surapprentissage plus important. Il serait donc intéressant d'explorer d'autres techniques pour contrôler le surapprentissage, comme la régularisation ou l'abandon (dropout). Les matrices de confusion fournies donnent également un aperçu des types d'erreurs commises par chaque modèle.

# 2. Classification par réseau multicouches avec convolution

# Conclusion

L'ajout de plus de neurones à la couche cachée augmente la capacité du modèle à apprendre des représentations plus complexes, mais cela peut également conduire à un surapprentissage si le modèle devient trop complexe par rapport aux données. Vous devriez donc observer attentivement l'évolution de la perte et de la précision sur l'ensemble de validation pour détecter d'éventuels signes de surapprentissage (par exemple, si la perte de validation commence à augmenter alors que la perte d'entraînement continue à diminuer).

l'ajout de plus de couches cachées permet au réseau de neurones d'apprendre des représentations plus complexes des données, mais cela peut également rendre l'entraînement plus difficile et augmenter le risque de surapprentissage. Il est donc important de surveiller attentivement l'évolution de la perte et de la précision sur l'ensemble de validation.

De plus, lorsque vous expérimentez avec différentes architectures, vous pouvez constater que les performances du modèle ne s'améliorent pas toujours lorsque vous ajoutez plus de couches ou de neurones. Cela est dû au fait que la capacité du modèle doit être adaptée à la complexité des données. Si le modèle est trop simple, il ne pourra pas apprendre les patterns dans les données (sous-apprentissage), mais s'il est trop complexe, il risque de mémoriser les données d'entraînement au lieu d'apprendre à généraliser à partir de celles-ci (surapprentissage).