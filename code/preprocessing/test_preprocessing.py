from PIL import Image
from utils import *
import shutil

# Pour les images de test dans le dataset initial
def create_folder_for_test_original(test):
    test_original = {}
    for emotion in range(8):
        class_data = test[test['emotion'] == emotion]
        test_original[emotion] = list(class_data['image'])
    
    path_folder = "code_ck+/test_images_folder/test_original"
    output_folder_test_original = Path(path_folder)
    if output_folder_test_original.exists():
        shutil.rmtree(output_folder_test_original)
    output_folder_test_original.mkdir(parents=True, exist_ok=True)

    for emotion, images in test_original.items():
        for i, image in enumerate(images):
            cv2.imwrite(str(output_folder_test_original / f"{emotion}_{i}.jpg"), image)

# Test des images des expériences
def process_image(image_path, output_folder, output_filename, convert_to_jpg=True):
    # Ouvrir l'image
    image = Image.open(image_path)
    # Convertir en niveaux de gris
    image = image.convert('L')
    # Redimensionner en 48x48 pixels
    image = image.resize((48, 48))
    # Sauvegarder l'image transformée dans le dossier de sortie si nécessaire
    if convert_to_jpg:
        output_path = os.path.join(output_folder, output_filename)
        image.save(output_path, 'JPEG')
    # Convertir l'image en tableau de pixels
    pixels = np.array(image).flatten()
    # Convertir le tableau en chaîne de caractères
    pixels_str = ' '.join(map(str, pixels))
    
    return image, pixels_str

# Créer un dossier pour les images de test des expériences
# folder_images_experience: chemin du dossier contenant les images de l'expérience
# (return) output_folder_test_experience: chemin du dossier de sortie pour les images de test
# return df_test_experience: DataFrame contenant les émotions, pixels et images de test
def create_folder_for_test_experience(folder_images_experience, output_folder_test_experience):
    # le supprimer et le recreer sinon pour être sûr de ne pas copier des images déjà existantes
    if os.path.exists(output_folder_test_experience):
        shutil.rmtree(output_folder_test_experience)
    os.makedirs(output_folder_test_experience, exist_ok=True)

    # Initialiser des listes pour stocker les données
    emotions = []
    pixel_test = []
    images = []

    # Parcourir le dossier des images
    for filename in os.listdir(folder_images_experience):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Chemin complet de l'image
            image_path = os.path.join(folder_images_experience, filename)
            # Extraire l'émotion du nom du fichier
            emotion = int(filename.split('_')[0])
            # Déterminer le nom de fichier de sortie en .jpg si l'entrée est .png ou .jpeg
            output_filename = filename.replace('.png', '.jpg').replace('.jpeg', '.jpg') 
            # Déterminer si l'image doit être convertie en .jpg
            convert_to_jpg = filename.endswith('.png') or filename.endswith('.jpeg')
            # Traiter l'image
            image, pixels_str = process_image(image_path, output_folder_test_experience, output_filename, convert_to_jpg=convert_to_jpg)   
            # Ajouter les données aux listes
            emotions.append(emotion)
            pixel_test.append(pixels_str)
            images.append(image)

    # Créer un DataFrame avec les colonnes 'emotion', 'pixels' et 'image'
    df_test_experience = pd.DataFrame({
        'emotion': emotions,
        'pixels': pixel_test,
        'image': images
    })

    return output_folder_test_experience, df_test_experience
