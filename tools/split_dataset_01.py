import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(data_dir, output_dir, train_size, test_size):
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    len_df = len(os.listdir(images_dir))
    proportion = int(len_df*0.1)
    print("Creating directories...")
    for split in tqdm(['train', 'val', 'test'], desc="Creating directories"):
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    filtered_labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    images = sorted(os.listdir(images_dir)[:proportion])
    labels = sorted(filtered_labels[:proportion])

    # split train, test
    initial_train_images, test_images, initial_train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    # split val from train
    val_size = 1.0 - train_size - test_size
    train_images, val_images, train_labels, val_labels = train_test_split(initial_train_images, initial_train_labels, test_size=val_size, random_state=42)

    print("\nCopying training set...")
    for img, lbl in tqdm(zip(train_images, train_labels), total=len(train_images), desc="Training set"):
        shutil.copy(os.path.join(images_dir, img), os.path.join(output_dir, 'train', 'images', img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(output_dir, 'train', 'labels', lbl))

    print("\nCopying validation set...")
    for img, lbl in tqdm(zip(val_images, val_labels), total=len(val_images), desc="Validation set"):
        shutil.copy(os.path.join(images_dir, img), os.path.join(output_dir, 'val', 'images', img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(output_dir, 'val', 'labels', lbl))

    print("\nCopying test set...")
    for img, lbl in tqdm(zip(test_images, test_labels), total=len(test_images), desc="Test set"):
        shutil.copy(os.path.join(images_dir, img), os.path.join(output_dir, 'test', 'images', img))
        shutil.copy(os.path.join(labels_dir, lbl), os.path.join(output_dir, 'test', 'labels', lbl))

    print(f"\nSplit dataset completed at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset")
    parser.add_argument('--data_dir', type=str, default="data/split_01/", required=True, help='Path to the data ')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output')
    parser.add_argument('--train_size', type=float, required=True, help='Proportion of train')
    parser.add_argument('--test_size', type=float, required=True, help='Proportion of test')

    args = parser.parse_args()
    split_dataset(args.data_dir, args.output_dir, args.train_size, args.test_size)