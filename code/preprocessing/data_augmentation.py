from utils import *

# Fonction pour appliquer le flou gaussien
def apply_gaussian_blur(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    return blurred_image

# Fonction pour appliquer la transformation affine
def apply_affine_transform(image):
    rows, cols = image.shape
    random_pts = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    new_pts = random_pts + np.random.normal(0, 0.1, random_pts.shape).astype(np.float32)
    M = cv2.getAffineTransform(random_pts, new_pts)
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    return transformed_image

# Fonction pour appliquer une transformation totale (translation + rotation)
def apply_total_transform(image):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    return transformed_image

# Fonction pour ajuster le contraste
def apply_contrast(image, alpha=1.5, beta=0):
    contrasted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrasted_image

# Fonction pour appliquer une transformation euclidienne (rotation)
def apply_euclidean_transform(image):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    transformed_image = cv2.warpAffine(image, M, (cols, rows))
    return transformed_image

# Fonction pour retourner l'image (flip)
def apply_flip(image):
    flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
    return flipped_image

# Fonction pour augmenter les données pour une classe spécifique
def augment_data(images, num_samples_per_class):
    augmented_images = []
    used_indices = set()  # Pour stocker les indices des images déjà utilisées
    while len(augmented_images) < num_samples_per_class:
        for idx, row in images.iterrows():
            if idx not in used_indices:
                image = row['image']
                
                # Appliquer les transformations avec une probabilité égale
                transformations = [
                    apply_gaussian_blur,
                    apply_affine_transform,
                    apply_total_transform,
                    apply_contrast,
                    apply_euclidean_transform,
                    apply_flip
                ]
                
                # Choisir aléatoirement une transformation
                transform = np.random.choice(transformations)
                augmented_image = transform(image)
                augmented_images.append(augmented_image)
                
                used_indices.add(idx)  # Ajouter l'indice utilisé
                
                if len(augmented_images) >= num_samples_per_class:
                    break
            else :
                used_indices = set()
        if len(augmented_images) >= num_samples_per_class:
            break

    return augmented_images
    
# Équilibrer les classes
# return output_folder_augmented_images: chemin du dossier contenant les images augmentées
def balance_classes(training, target_samples_per_class):  
    augmented_data = {}
    for emotion in range(8):
        class_data = training[training['emotion'] == emotion]
        num_samples = target_samples_per_class - len(class_data)
        if num_samples > 0:
            augmented_images = augment_data(class_data, num_samples)
            augmented_data[emotion] = list(class_data['image']) + augmented_images
        else:
            augmented_data[emotion] = list(class_data['image'])

    # Enregistrer les images augmentées dans un dossier
    output_folder_augmented_images = Path('code_ck+/augmented_images')
    if output_folder_augmented_images.exists():
        # delete it
        shutil.rmtree(output_folder_augmented_images)
        output_folder_augmented_images.mkdir()
    else : 
        output_folder_augmented_images.mkdir(parents=True, exist_ok=True)

    for emotion, images in augmented_data.items():
        for i, image in enumerate(images):
            cv2.imwrite(str(output_folder_augmented_images / f"{emotion}_{i}.jpg"), image)
            
    return output_folder_augmented_images