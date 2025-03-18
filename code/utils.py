import pandas as pd  # Pandas for data manipulation and analysis
import numpy as np  # NumPy for numerical operations
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import shutil
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import random
import joblib

def display_images_from_folder(folder_path, num_images=5):
    folder = Path(folder_path)
    image_files = list(folder.glob("*.jpg"))

    # Limiter le nombre d'images à afficher
    image_files = image_files[:num_images]

    plt.figure(figsize=(15, 5))

    for idx, image_file in enumerate(image_files):
        img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, num_images, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Image {idx}")
        plt.axis("off")

    plt.show()


def count_images_in_folder(folder_path):
    folder = Path(folder_path)
    image_files = list(folder.glob("*"))
    return len(image_files)


def set_seed(seed):
    # Fixer la graine pour le module random
    random.seed(seed)

    # Fixer la graine pour NumPy
    np.random.seed(seed)


# Afficher les images bien classées
def display_correctly_classified_images(y_test, y_pred, original_images, filenames):
    # Dictionnaire pour stocker une image bien classée par classe (0 à 7)
    correct_images = {}

    # Parcours des indices des images bien classées pour chaque classe
    for idx in np.where(y_pred == y_test)[
        0
    ]:  # Indices des images correctement classées
        true_label = y_test[idx]
        if true_label not in correct_images and 0 <= true_label <= 7:
            # Si la classe n'a pas encore été ajoutée au dictionnaire, on ajoute cette image
            correct_images[true_label] = {
                "image": original_images[idx],
                "true_label": true_label,
                "pred_label": y_pred[idx],
                "filename": filenames[idx],
            }

        # Si nous avons trouvé des images pour toutes les classes de 0 à 7, on peut arrêter la boucle
        if len(correct_images) == 8:
            break

    # Affichage des images bien classées pour chaque classe (0 à 7)
    plt.figure(figsize=(10, 5))
    for i in range(8):
        if i in correct_images:
            data = correct_images[i]
            image = data["image"]
            true_label = data["true_label"]
            # pred_label = data["pred_label"]
            filename = data["filename"]

            plt.subplot(1, 8, i + 1)
            plt.imshow(image, cmap="gray")
            plt.title(f"Classe {true_label}\nNom: {filename}")
            plt.axis("off")
        else:
            # S'il n'y a pas d'image pour cette classe, laisser la case vide
            plt.subplot(1, 8, i + 1)
            plt.axis("off")
            plt.title(f"Classe {i}\nPas d'image")

    plt.tight_layout()
    plt.show()


def save_variables(name_file, **kwargs):
    # Enregistrer les variables dans un fichier
    joblib.dump(kwargs, name_file + ".joblib")

def afficher_matrice_confusion(conf_matrix_normalized):
# Créer une figure pour la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[str(i) for i in range(8)],
        yticklabels=[str(i) for i in range(8)],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix with Percentages")
    plt.show()
    
def load_variables(name_file, **kwargs):
    # Charger les variables à partir du fichier
    variables = joblib.load(name_file + ".joblib")

    # Extraire les variables en utilisant les noms dynamiques
    loaded_vars = {key: variables[key] for key in kwargs.values()}

    return loaded_vars

def load_results_svm(extracteur, kernel, augmented):
    suffix = '_aug' if augmented else ''
    name_accuracy_score = "accuracy_score"
    name_y_pred = "predictions"
    name_matrice = "confusion_matrix"
    name_report = "classification_report"    
    
    # Charger les variables en fonction de l'extracteur et du kernel
    if extracteur == "VGG16":
        variables = load_variables(f'accuracy_vgg_svm{suffix}_{kernel}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report)
    elif extracteur == "MobileNet":
        variables = load_variables(f'accuracy_mobilenet_svm{suffix}_{kernel}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report)
    elif extracteur == "HOG":
        variables = load_variables(f'accuracy_hog_svm{suffix}_{kernel}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report)
    elif extracteur == "Pixels":
        variables = load_variables(f'accuracy_pixels_svm{suffix}_{kernel}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report)
    else:
        raise ValueError("Extracteur non reconnu")

    # Print accuracy
    print(f"Accuracy: {variables['accuracy_score']:.2f}%")
    # Print predictions
    print(f"Predictions: {variables['predictions']}")
    # Afficher la matrice de confusion
    afficher_matrice_confusion(variables['confusion_matrix'])

    # Print le rapport
    print(variables['classification_report'])
    
def load_results_knn(extracteur, augmented):
    suffix = '_aug' if augmented else ''
    name_accuracy_score = "accuracy_score"
    name_y_pred = "predictions"
    name_matrice = "confusion_matrix"
    name_report = "classification_report" 
    name_best_n_neighbors = "best_n_neighbors"   
    
    # Charger les variables en fonction de l'extracteur et du kernel
    if extracteur == "VGG16":
        variables = load_variables(f'accuracy_vgg_knn{suffix}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report, best_n_neighbors=name_best_n_neighbors)
    elif extracteur == "MobileNet":
        variables = load_variables(f'accuracy_mobilenet_knn{suffix}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report, best_n_neighbors=name_best_n_neighbors)
    elif extracteur == "HOG":
        variables = load_variables(f'accuracy_hog_knn{suffix}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report, best_n_neighbors=name_best_n_neighbors)
    elif extracteur == "Pixels":
        variables = load_variables(f'accuracy_pixels_knn{suffix}', accuracy=name_accuracy_score, pred=name_y_pred, matrice=name_matrice, report=name_report, best_n_neighbors=name_best_n_neighbors)
    else:
        raise ValueError("Extracteur non reconnu")
    # Print accuracy
    print(f"Accuracy: {variables['accuracy_score']:.2f}%")
    # Print predictions
    print(f"Predictions: {variables['predictions']}")
    # Print best n_neighbors
    print(f"Best n neighbors: {variables['best_n_neighbors']}")
    # Afficher la matrice de confusion
    afficher_matrice_confusion(variables['confusion_matrix'])

    # Print le rapport
    print(variables['classification_report'])

def choisir_kernel():
    print("Choisissez un kernel parmi les options suivantes:")
    print("1. linear")
    print("2. rbf")
    print("3. poly")
    choix_kernel = int(input("Entrez le numéro du kernel: "))

    # Map des kernels
    kernels = {1: "linear", 2: "rbf", 3: "poly"}

    if choix_kernel not in kernels:
        raise ValueError("Choix de kernel invalide")
    kernel = kernels[choix_kernel]
    
    return kernel
def choisir_extracteur(isSVM):
    # Choix de l'extracteur
    print("Choisissez un extracteur parmi les options suivantes:")
    print("1. HOG")
    print("2. Pixels")
    print("3. MobileNet")
    print("4. VGG")
    choix_extracteur = int(input("Entrez le numéro de l'extracteur: "))

    # Map des extracteurs
    extracteurs = {1: "HOG", 2: "Pixels", 3: "MobileNet", 4: "VGG"}

    if choix_extracteur not in extracteurs:
        raise ValueError("Choix d'extracteur invalide")

    extracteur = extracteurs[choix_extracteur]

    # Choix du kernel
    if isSVM : 
        kernel = choisir_kernel()

    # Choix de l'augmentation de données
    print("Souhaitez-vous appliquer une augmentation de données ?")
    print("1. Oui")
    print("2. Non")
    choix_augmentation = int(input("Entrez le numéro de votre choix: "))

    augmentation = False
    if choix_augmentation == 1:
        augmentation = True
    elif choix_augmentation != 2:
        raise ValueError("Choix d'augmentation invalide")

    if isSVM:
        return extracteur, kernel, augmentation
    else:
        return extracteur, augmentation