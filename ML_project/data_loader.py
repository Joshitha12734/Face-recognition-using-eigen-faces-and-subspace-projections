
import numpy as np
from PIL import Image
import os
import re

def load_yale_dataset(path):
    """
    Load Yale Face Database from folder containing subjectXX.* files
    
    Args:
        path: Path to folder with all .gif images
        
    Returns:
        X: numpy array of shape (n_images, n_pixels)
        y: numpy array of shape (n_images,) with subject IDs (0-14)
    """
    images = []
    labels = []
    
    image_files = [f for f in os.listdir(path) if f.endswith(('.gif', '.GIF'))]
    
    print(f"Found {len(image_files)} image files")
    
    
    pattern = r'subject(\d+)\.'
    
    for img_file in image_files:
        
        match = re.search(pattern, img_file)
        if match:
            subject_id = int(match.group(1)) - 1  # 0-indexed (0 to 14)
            img_path = os.path.join(path, img_file)
            
            try:
               
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img, dtype=np.float32)
                
                images.append(img_array.flatten())
                labels.append(subject_id)
                
            except PermissionError:
                print(f"Skipping {img_file} (permission denied)")
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    
    X = np.array(images)  # Shape: (n_images, n_pixels)
    y = np.array(labels)
    
    print(f"Successfully loaded {X.shape[0]} images")
    print(f"Image size: {X.shape[1]} pixels")
    print(f"Number of subjects: {len(np.unique(y))}")
    
    return X, y

if __name__ == "__main__":
    path = r"C:\Users\MURALI KRISHNA\Downloads\archive"
    X, y = load_yale_dataset(path)
    print("\n" + "="*50)
    print("DATASET LOADED SUCCESSFULLY!")
    print("="*50)
    print(f"Data shape: {X.shape}")
    print(f"Unique people: {len(set(y))}")
    print(f"Pixel range: [{X.min():.1f}, {X.max():.1f}]")

    print(f"\nImages per subject:")
    for i in range(15):
        count = np.sum(y == i)
        print(f"  Subject {i+1}: {count} images")