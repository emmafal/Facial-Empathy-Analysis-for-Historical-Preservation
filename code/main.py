from preprocessing import (
    data_augmentation as da,
    nettoyage_formatage_data as nfd,
    test_preprocessing as tp,
)
from extracteurs_caracteristiques.hog import (
    extractor_train as hog_extractor_train,
    extractor_test as hog_extractor_test,
)
from extracteurs_caracteristiques.mobileNet import (
    extractor_train as mobileNet_extractor_train,
    extractor_test as mobileNet_extractor_test,
)
from extracteurs_caracteristiques.pixels import (
    extractor_train as pixels_extractor_train,
    extractor_test as pixels_extractor_test,
)
from extracteurs_caracteristiques.vgg import (
    extractor_train as vgg_extractor_train,
    extractor_test as vgg_extractor_test,
)
from model import svm, knn
from utils import *

# Fixer la graine à 42
set_seed(42)

# Charger le dataset initial
ckpixelset = pd.read_csv("/home/emma/Data/UTC/cours/GI04/TX/ckextended.csv")
train, test = nfd.train_test_data_split(ckpixelset)

# # Créer un dossier pour les images de test inclus dans le dataset initial
tp.create_folder_for_test_original(test)

# # Créer un dossier pour les images de test des expériences
tp.create_folder_for_test_experience(
    "code_ck+/test_images_folder/test_original",
    "code_ck+/test_images_folder/test_experience_processed",
)

# # Faire une augmentation des données de train pour équilibrer les classes
output_folder_augmented_images = da.balance_classes(train, 1000)

# Afficher les images d'un dossier
# tp.display_images_from_folder(output_folder_augmented_images, num_images=5)

# Compte le nombre d'images dans un dossier
# num_images = count_images_in_folder(output_folder_augmented_images)

# UTILISER SVM SANS EXTRACTION DE CARACTERISTIQUES
# Training et test sans extracteur ni augmentation de données
# X_train, y_train, X_test, y_test, original_images, filenames = svm.without_anything(
#     "code_ck+/test_images_folder/test_original", train, test
# )  # 81% fonctionne
# accuracy, y_pred = svm.svm_train_without(X_train, y_train, X_test, y_test, augmentation=False)
# conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(8))
# # Normaliser la matrice de confusion pour obtenir des pourcentages
# matrice = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
# print(matrice)

# UTILISER SVM SANS EXTRACTION DE CARACTERISTIQUES
# Training et test sans extracteur mais avec augmentation de données
# pas forcément utile car mauvais résultats sans augmentation

# Training SVM AVEC EXTRACTION DE CARACTERISTIQUES
# extracteur, kernel, augmentation = choisir_extracteur(True)
# if augmentation:
#     accuracy_score, y_pred, matrice, report = svm.entrainer_avec_augmentation(extracteur, kernel)
# else:
#     accuracy_score, y_pred, matrice, report = svm.entrainer_sans_augmentation(train, extracteur, kernel)

# # Charger les résultats
extracteur, kernel, augmentation = choisir_extracteur(True)
load_results_svm(extracteur, kernel, augmentation)

# --------------------------------------------
# Utiliser KNN
# Training et test sans extracteur ni augmentation de données
# X_train, y_train, X_test, y_test, original_images, filenames = knn.without_anything(
#     'code_ck+/test_images_folder/test_original', train, test
# )

# Training KNN AVEC EXTRACTION DE CARACTERISTIQUES
# extracteur, augmentation = choisir_extracteur(False)
# if augmentation:
#     accuracy_score, y_pred, matrice, report, best_n_neighbors = knn.entrainer_avec_augmentation(extracteur)
# else:
#     accuracy_score, y_pred, matrice, report, best_n_neighbors = knn.entrainer_sans_augmentation(train, extracteur)

# Charger les résultats
# extracteur, augmentation = choisir_extracteur(False)
# load_results_knn(extracteur, augmentation)