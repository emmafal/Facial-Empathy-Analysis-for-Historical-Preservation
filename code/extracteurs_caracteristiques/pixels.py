from utils import *

# Fonction pour extraire les caractéristiques Pixels des images
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

    # Aplatir les images pour obtenir des vecteurs de pixels
    pixel_features = [image.flatten() for image in data]
    pixel_features = np.array(pixel_features)

    # Normaliser les caractéristiques #todo sans normalisation
    scaler = StandardScaler()
    pixel_features = scaler.fit_transform(pixel_features)

    # Diviser les données en ensembles d'entraînement et de test
    X_train= pixel_features
    y_train = labels
    return X_train, y_train, scaler

# Fonction pour extraire les caractéristiques Pixels des images de test
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
    
    # Aplatir les images pour obtenir des vecteurs de pixels
    pixel_features = [image.flatten() for image in data]
    pixel_features = np.array(pixel_features)

    # Normaliser les caractéristiques
    pixel_features = scaler.transform(pixel_features)

    X_test= pixel_features
    y_test = labels
    return X_test, y_test, data, filenames # images d'origine