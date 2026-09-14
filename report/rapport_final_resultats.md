# Sections de résultats générées automatiquement

Ces tableaux doivent être copiés dans le rapport Word final après l'exécution des scripts.

## Partie I - MLP / données tabulaires

| model | initialization | accuracy | precision_macro | recall_macro | f1_macro | best_val_accuracy | trainable_parameters | device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequential | gaussian | 0.8684 | 0.8642 | 0.8531 | 0.8579 | 0.9211 | 4130.00 | cpu |
| sequential | constant | 0.8772 | 0.8720 | 0.8647 | 0.8681 | 0.9211 | 4130.00 | cpu |
| sequential | xavier | 0.7632 | 0.7486 | 0.7411 | 0.7442 | 0.7719 | 4130.00 | cpu |
| custom_class | gaussian | 0.8158 | 0.8859 | 0.7558 | 0.7740 | 0.8421 | 4130.00 | cpu |
| custom_class | constant | 0.8860 | 0.8834 | 0.8718 | 0.8769 | 0.9211 | 4130.00 | cpu |
| custom_class | xavier | 0.7895 | 0.7796 | 0.7943 | 0.7828 | 0.8509 | 4130.00 | cpu |

Analyse attendue : comparer nn.Sequential et classe personnalisée, puis expliquer l'impact des initialisations gaussienne, constante et Xavier sur la stabilité de l'apprentissage.

## Partie II - CNN / images

_Résultats non disponibles. Exécuter le script correspondant._

Analyse attendue : comparer MLP et CNN, puis interpréter l'effet du padding, du pooling, du nombre de filtres et de la convolution 1x1.

## Partie III - RNN / LSTM / GRU

_Résultats non disponibles. Exécuter le script correspondant._

Analyse attendue : discuter stabilité, capacité mémoire, gradient clipping, coût de calcul et performance relative de RNN, LSTM et GRU.

## Mini Seq2Seq

| source | reference | greedy | beam_search | bleu1_greedy | bleu1_beam |
| --- | --- | --- | --- | --- | --- |
| i am a student | je suis un etudiant | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.0000 | 0.0000 |
| he is a teacher | il est un professeur | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.0000 | 0.0000 |
| she likes music | elle aime la musique | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.1111 | 0.1111 |
| we study deep learning | nous etudions le deep learning | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.0000 | 0.0000 |
| the model is training | le modele est en entrainement | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.0000 | 0.0000 |
| the data is clean | les donnees sont propres | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.1111 | 0.1111 |
| i like artificial intelligence | j aime l intelligence artificielle | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.0000 | 0.0000 |
| this result is good | ce resultat est bon | la donnees la la donnees la la donnees la | la la donnees la la donnees la la donnees | 0.0000 | 0.0000 |
