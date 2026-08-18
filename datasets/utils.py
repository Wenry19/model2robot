
import os

def get_images_paths_and_labels(target_path, class_names):

    img_paths = []
    labels = []

    for i, c in enumerate(class_names):
        class_path = os.path.join(target_path, c)
        curr_img_paths = [os.path.join(class_path, img_name)
                          for img_name in os.listdir(class_path)]
        curr_labels = [i] * len(curr_img_paths)

        img_paths += curr_img_paths
        labels += curr_labels
    
    return img_paths, labels