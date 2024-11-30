import os
from torchvision import datasets

# Path to save images
output_folder = "measure/CIFAR10"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Download the CIFAR-10 dataset
cifar10 = datasets.CIFAR10(root="cifar10_data", train=True, download=True)

# Save images as JPG in folders named after their classes
for idx, (image, label) in enumerate(cifar10):
    class_name = cifar10.classes[label]
    #class_folder = os.path.join(output_folder, class_name)
    class_folder = output_folder
    os.makedirs(class_folder, exist_ok=True)  # Create a folder for the class if it doesn't exist

    # Save the image directly as JPG
    file_path = os.path.join(class_folder, f"{idx}.jpg")
    image.save(file_path, format="JPEG")

    if (idx + 1) % 1000 == 0:
        print(f"Saved {idx + 1}/{len(cifar10)} images.")

print(f"All CIFAR-10 images saved in '{output_folder}' folder.")
