from utils import *


# Fonction pour extraire les caractéristiques des images
def extractor_train(training, isAugmented, batch_size=100):
    # Charger le modèle MobileNet pré-entraîné sans les couches de classification finales
    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    model = Model(inputs=base_model.input, outputs=base_model.output)
    if isAugmented:
        # Variables pour stocker les données et les étiquettes
        data = []
        labels = []
        features = []

        # Charger les images et les étiquettes
        images = []  # Pour accumuler les images dans un batch temporaire
        temp_labels = []  # Pour accumuler les labels des images
        for filename in os.listdir(training):
            if filename.endswith(".jpg"):
                filepath = os.path.join(training, filename)
                image = cv2.imread(filepath)
                image = cv2.resize(
                    image, (224, 224)
                )  # Redimensionner l'image à la taille requise par MobileNet
                image = img_to_array(image)
                img = np.array(image, dtype="float32")
                img = preprocess_input(img)
                print('image.append juste après')
                images.append(img)

                # Stocker le label correspondant
                label = int(
                    filename.split("_")[0]
                )  # L'étiquette est le premier élément du nom du fichier
                temp_labels.append(label)
                
                print('image', len(images))
                # Si on atteint la taille du batch, on effectue une prédiction
                if len(images) == batch_size:
                    print("batch rempli")
                    # Convertir en tableau NumPy et passer au modèle
                    batch_images = np.array(images)
                    batch_features = model.predict(batch_images, verbose=0)
                    print('prediction faite')
                    # Aplatir les caractéristiques pour chaque image du batch
                    batch_features = batch_features.reshape(
                        (batch_features.shape[0], -1)
                    )
                    features.extend(batch_features)
                    labels.extend(temp_labels)

                    # Réinitialiser les accumulateurs pour le prochain batch
                    images = []
                    temp_labels = []

        # Si des images restent après la boucle (moins d'un batch complet), les traiter
        if len(images) > 0:
            batch_images = np.array(images)
            batch_features = model.predict(batch_images, verbose=0)
            batch_features = batch_features.reshape((batch_features.shape[0], -1))
            features.extend(batch_features)
            labels.extend(temp_labels)

        features = np.array(features)
        labels = np.array(labels)
        # Normaliser les caractéristiques
        print('normalisation')
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X_train = features
        y_train = labels

    else:
        # Convertir les données en array NumPy et prétraiter les images
        labels = training["emotion"]
        data = []
        for image in training["image"]:
            image = cv2.cvtColor(
                image, cv2.COLOR_GRAY2BGR
            )  # Convertir l'image en couleur
            image = cv2.resize(
                image, (224, 224)
            )  # Redimensionner l'image à la taille requise par MobileNet
            image = img_to_array(image)
            data.append(image)

        # Convertir les étiquettes en array NumPy
        labels = np.array(labels)

        # Convertir les données en array NumPy et prétraiter les images
        data = np.array(data, dtype="float32")
        data = preprocess_input(data)

        # Extraire les caractéristiques des images
        features = model.predict(data)
        features = features.reshape((features.shape[0], -1))

        # Normaliser les caractéristiques
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X_train = features
        y_train = labels

    return X_train, y_train, scaler


# Fonction pour extraire les caractéristiques Pixels des images de test
def extractor_test(test_folder, scaler):
    data = []
    labels = []
    filenames = []

    # Charger les images et les étiquettes
    for filename in os.listdir(test_folder):
        if filename.endswith(".jpg"):
            filepath = os.path.join(test_folder, filename)
            image = cv2.imread(filepath)
            image = cv2.resize(
                image, (224, 224)
            )  # Redimensionner l'image à la taille requise par MobileNet
            image = img_to_array(image)
            data.append(image)
            label = int(
                filename.split("_")[0]
            )  # L'étiquette est le premier élément du nom du fichier
            labels.append(label)
            filenames.append(filename)

    # Convertir les étiquettes en array NumPy
    labels = np.array(labels)

    # Convertir les données en array NumPy et prétraiter les images
    data = np.array(data, dtype="float32")
    data = preprocess_input(data)

    # Charger le modèle MobileNet pré-entraîné sans les couches de classification finales
    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Extraire les caractéristiques des images
    features = model.predict(data)
    features = features.reshape((features.shape[0], -1))

    # Normaliser les caractéristiques
    features = scaler.transform(features)

    X_test = features
    y_test = labels
    return X_test, y_test, data, filenames
