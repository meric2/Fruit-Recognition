import os
from PIL import Image

def rename_images(folder_path, fruit_name ):
    # Ensure the folder path ends with a '/'
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    folder_path+= fruit_name + '/'

    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter out only image files (you may want to extend the list of image file extensions)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Sort the image files to maintain order
    image_files.sort()

    # Rename and save each image
    for index, file_name in enumerate(image_files, start=1):
        original_path = os.path.join(folder_path, file_name)

        # Generate the new file name
        new_file_name = fruit_name + "_" + str(index) + ".jpg"
        new_path = os.path.join(folder_path, new_file_name)

        # Rename and save the image
        os.rename(original_path, new_path)

if __name__ == "__main__":
    folder_path = "data/"
    
    if os.path.isdir(folder_path):
        rename_images(folder_path, "cherry")
        print("Images renamed successfully.")
    else:
        print("Invalid folder path. Please provide a valid path.")
