from utils import *
from extracteurs_caracteristiques import hog
from extracteurs_caracteristiques import pixels
from extracteurs_caracteristiques import mobileNet
from extracteurs_caracteristiques import vgg


# Sans extracteur et sans augmentation de données
def without_anything(test_folder, training_without_augmentation, df_test):
    X_train = np.array(
        [
            np.fromstring(pixels, dtype=int, sep=" ")
            for pixels in training_without_augmentation["pixels"]
        ]
    )
    y_train = np.array(training_without_augmentation["emotion"])
    # Test données initiales
    X_test = np.array(
        [np.fromstring(pixels, dtype=int, sep=" ") for pixels in df_test["pixels"]]
    )
    data = []
    filenames = []
    labels = []
    # Charger les images et les étiquettes
    for filename in os.listdir(test_folder):
        if filename.endswith(".jpg"):
            filepath = os.path.join(test_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            label = int(
                filename.split("_")[0]
            )  # L'étiquette est le premier élément du nom du fichier
            data.append(image)
            labels.append(label)
            filenames.append(filename)
    y_test = np.array(labels)

    return X_train, y_train, X_test, y_test, data, filenames


# Avec extracteur et sans augmentation de données
def with_extractor_only(
    training_without_augmentation, test_folder, extractor_train, extractor_test
):
    X_train, y_train, scaler = extractor_train(training_without_augmentation, False)
    X_test, y_test, original_images, filenames = extractor_test(test_folder, scaler)
    return X_train, y_train, X_test, y_test, original_images, filenames, scaler


# Avec extracteur et augmentation de données
def with_extractor_and_data_augmentation(
    training_augmented, test_folder, extractor_train, extractor_test
):
    X_train, y_train, scaler = extractor_train(training_augmented, True)
    X_test, y_test, original_images, filenames = extractor_test(test_folder, scaler)
    return X_train, y_train, X_test, y_test, original_images, filenames, scaler


# Entrainer le modèle KNN
def knn_train(X_train, y_train, X_test, y_test, extracteur, augmentation):
    # De 1 à 500 voisins (exclu), échelle logarithmique
    n_neighbors_list = np.unique(np.round(np.geomspace(1, 500, 100)).astype(int))
    param_grid = {"n_neighbors": n_neighbors_list}
    cls = KNeighborsClassifier()
    search = GridSearchCV(cls, param_grid, scoring="accuracy", cv=10)
    search.fit(X_train, y_train)

    # Entraîner le classificateur KNN avec les meilleurs paramètres
    best_n_neighbors = search.best_params_["n_neighbors"]
    knn_model = KNeighborsClassifier(
        n_neighbors=best_n_neighbors
    )  
    # Vous pouvez ajuster le nombre de voisins (k)
    knn_model.fit(X_train, y_train)
    # Enregistrer le modèle
    suffixe = "_aug" if augmentation else ""
    joblib.dump(knn_model, f"knn_{extracteur.lower()}_{suffixe}.pkl")
    # TODO verifier que c'est bien le modèle qui est enregistré pour pouvoir l'utiliser dans test_lime_entier.py
    print(type(knn_model)) 
    # Prédire les étiquettes des données de test
    y_pred = knn_model.predict(X_test)

    # Calculer l'exactitude
    accuracy = accuracy_score(y_test, y_pred) * 100

    return accuracy, y_pred, best_n_neighbors


def entrainer_avec_augmentation(extracteur):
    # Dictionnaire des extracteurs et des noyaux pour l'augmentation de données
    extracteurs_data_augmentation = {
        "HOG": (
            with_extractor_and_data_augmentation,
            hog.extractor_train,
            hog.extractor_test,
        ),
        "Pixels": (
            with_extractor_and_data_augmentation,
            pixels.extractor_train,
            pixels.extractor_test,
        ),
        "MobileNet": (
            with_extractor_and_data_augmentation,
            mobileNet.extractor_train,
            mobileNet.extractor_test,
        ),
        "VGG": (
            with_extractor_and_data_augmentation,
            vgg.extractor_train,
            vgg.extractor_test,
        ),
    }

    # Vérification si l'extracteur choisi existe dans les données
    if extracteur not in extracteurs_data_augmentation:
        raise ValueError(f"Extracteur {extracteur} non trouvé.")

    # Appel de la fonction d'augmentation de données
    augmentation_function, train_extractor, test_extractor = (
        extracteurs_data_augmentation[extracteur]
    )

    # Appel de l'extracteur avec augmentation
    X_train, y_train, X_test, y_test, original_images, filenames, scaler = (
        augmentation_function(
            "code_ck+/augmented_images",
            "code_ck+/test_images_folder/test_original",
            train_extractor,
            test_extractor,
        )
    )

    # Entraîner et sauvegarder les résultats
    accuracy_score, y_pred, matrice, report, best_n_neighbors = (
        entrainer_et_resultats_knn(
            X_train,
            y_train,
            X_test,
            y_test,
            extracteur,
            True,
            original_images,
            filenames,
            scaler,
        )
    )

    return accuracy_score, y_pred, matrice, report, best_n_neighbors


def entrainer_sans_augmentation(train, extracteur):
    # Dictionnaire des extracteurs et des noyaux sans augmentation de données
    extracteurs_data = {
        "HOG": (with_extractor_only, hog.extractor_train, hog.extractor_test),
        "Pixels": (with_extractor_only, pixels.extractor_train, pixels.extractor_test),
        "MobileNet": (
            with_extractor_only,
            mobileNet.extractor_train,
            mobileNet.extractor_test,
        ),
        "VGG": (with_extractor_only, vgg.extractor_train, vgg.extractor_test),
    }

    # Vérification si l'extracteur choisi existe dans les données
    if extracteur not in extracteurs_data:
        raise ValueError(f"Extracteur {extracteur} non trouvé.")

    # Appel de la fonction sans augmentation de données
    extraction_function, train_extractor, test_extractor = extracteurs_data[extracteur]

    # Appel de l'extracteur sans augmentation
    print("Extraction des caractéristiques...")
    # TODO mettre scaler partout extracteurs
    X_train, y_train, X_test, y_test, original_images, filenames, scaler = extraction_function(
        train,
        "code_ck+/test_images_folder/test_original",
        train_extractor,
        test_extractor,
    )

    # Entraîner et sauvegarder les résultats
    print("Entraînement du modèle KNN...")
    accuracy_score, y_pred, matrice, report, best_n_neighbors = (
        entrainer_et_resultats_knn(
            X_train,
            y_train,
            X_test,
            y_test,
            extracteur,
            False,
            original_images,
            filenames,
            scaler,
        )
    )

    return accuracy_score, y_pred, matrice, report, best_n_neighbors


def entrainer_et_resultats_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    extracteur,
    augmentation,
    original_images,
    filenames,
    scaler,
):
    # Entraîner le modèle KNN
    accuracy_score, y_pred, best_n_neighbors = knn_train(
        X_train, y_train, X_test, y_test, extracteur, augmentation
    )

    # Générer la matrice de confusion et le rapport de classification
    conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(8))
    # Normaliser la matrice de confusion pour obtenir des pourcentages
    matrice = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    report = classification_report(y_test, y_pred)

    # Sauvegarder les résultats
    print("Sauvegarde des résultats...")
    suffixe = "_aug" if augmentation else ""
    save_variables(
        f"accuracy_{extracteur.lower()}_knn{suffixe}",
        accuracy_score=accuracy_score,
        predictions=y_pred,
        confusion_matrix=matrice,
        classification_report=report,
        best_n_neighbors=best_n_neighbors,
        scaler = scaler,
    )

    # Affichage des résultats
    print(f"Accuracy: {accuracy_score}")
    print(f"Predictions: {y_pred}")
    print(f"Classification Report:\n{report}")
    print("Best n neighbors: ", best_n_neighbors)
    print("-" * 40)

    # Affichage des images correctement classées
    display_correctly_classified_images(y_test, y_pred, original_images, filenames)

    return accuracy_score, y_pred, matrice, report, best_n_neighbors
