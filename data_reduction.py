"""
datasetteki resim sayısını azaltmak için kullanılır.
fazla resimleri başka bir klasöre taşır.
"""


import os
import random
import shutil

# Klasör yollarını belirtin
dataset_path = "venv/fruit_dataset"
remains_path = "venv/remains_images"

# Her sınıf için işlem yapın
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    
    # Sınıf klasöründeki resimleri listele
    images = os.listdir(class_path)
    
    # Eğer resim sayısı 500'den fazlaysa, rastgele 500 resim seçin
    if len(images) > 500:
        random.shuffle(images)
        images_to_remove = images[500:]
        
        # Silinecek resimleri taşıyın
        for image in images_to_remove:
            image_path = os.path.join(class_path, image)
            remains_image_path = os.path.join(remains_path, class_folder, image)
            os.makedirs(os.path.dirname(remains_image_path), exist_ok=True)
            shutil.move(image_path, remains_image_path)