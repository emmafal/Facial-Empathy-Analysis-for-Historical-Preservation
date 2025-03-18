from utils import *

# Fonction pour charger et transformer les pixels en images
def pixels_to_image(row, image_size=(48, 48)):
    # Vérifier si 'pixels' est déjà un tableau NumPy
    if isinstance(row['pixels'], np.ndarray):
        pixels = row['pixels']
    else:
        # Supposer que 'pixels' est une chaîne de caractères et la diviser
        pixels = np.array(row['pixels'].split(), dtype='uint8')
    image = pixels.reshape(image_size)
    return image

# Charger le dataset initial et transformer les pixels en images
def charge_and_transform_dataset(dataset, catégorie=['Training', 'PublicTest', 'PrivateTest']):
    dataset = dataset[dataset['Usage'].isin(catégorie)].copy()
    dataset.drop('Usage', axis=1, inplace=True)
    dataset.reset_index(drop=True, inplace=True) 
    dataset['image'] = dataset.apply(pixels_to_image, axis=1) # Transformer les pixels en images
    return dataset

# Diviser le dataset initial en données d'entraînement et de test
# Rassembler les données de test en un seul ensemble pour maximiser les données de test
def train_test_data_split(dataset):
    training = charge_and_transform_dataset(dataset, catégorie=['Training'])
    # On assemble les 2 tests en un seul pour maximiser les données de test
    test = charge_and_transform_dataset(dataset, catégorie=['PublicTest', 'PrivateTest'])
    return training, test

