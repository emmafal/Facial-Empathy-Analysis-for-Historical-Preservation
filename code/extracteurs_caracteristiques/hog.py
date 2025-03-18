from utils import *

# Fonction pour extraire les caractéristiques HOG des images
def extractor_train(training, isAugmented): 
    if isAugmented:
        # Variables pour stocker les données et les étiquettes
        data = []
        labels = []
        # Charger les images et les étiquettes
        for filename in os.listdir(training):
            if filename.endswith('.jpg'):
                filepath = os.path.join(training, filename)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                label = int(filename.split('_')[0])  # L'étiquette est le premier élément du nom du fichier
                data.append(image)
                labels.append(label)
    else:
        labels = training['emotion']
        data = training['image']

    # Convertir les étiquettes en array NumPy
    labels = np.array(labels)

    # Extraire les caractéristiques HOG des images
    hog_features = []
    for image in data:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)

    hog_features = np.array(hog_features)

    # Normaliser les caractéristiques
    scaler = StandardScaler()
    hog_features = scaler.fit_transform(hog_features)

    # Diviser les données en ensembles d'entraînement et de test
    X_train = hog_features
    y_train = labels
    
    return X_train, y_train, scaler

# Fonction pour extraire les caractéristiques HOG des images de test
def extractor_test(test_folder, scaler):
    data = []
    labels = []
    filenames = []
    
    # Charger les images et les étiquettes
    for filename in os.listdir(test_folder):
        if filename.endswith('.jpg'):
            filepath = os.path.join(test_folder, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            label = int(filename.split('_')[0])  # L'étiquette est le premier élément du nom du fichier
            data.append(image)
            labels.append(label)
            filenames.append(filename)

    # Convertir les étiquettes en array NumPy
    labels = np.array(labels)

    # Extraire les caractéristiques HOG des images
    hog_features = []
    for image in data:
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    hog_features = np.array(hog_features)
    
    # Normaliser les caractéristiques par rapport à l'ensemble d'entraînement
    hog_features = scaler.transform(hog_features)

    X_test = hog_features
    y_test = labels
    return X_test, y_test, data, filenames