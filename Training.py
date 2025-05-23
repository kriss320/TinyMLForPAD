import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import timm
import numpy as np
import time

timestr = time.strftime("%Y-%m-%d")


# Dataset Class Definition
#https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    def __init__(self, training_dataset_path, transform=None, target_transform=None):
        self.training_dataset_path = training_dataset_path
        self.transform = transform
        self.target_transform = target_transform
        self.img_classifications = []
        self.classes = sorted(os.listdir(training_dataset_path))
        # the dataset expects the images to be in folders named after the class. 
        for classification, class_name in enumerate(self.classes):
            class_dir = os.path.join(training_dataset_path, class_name)
            for root, dirs, files in os.walk(class_dir):
                for img_file in files:
                    img_path = os.path.join(root, img_file)
                    self.img_classifications.append((img_path, classification))
        np.random.shuffle(self.img_classifications) #shuffle the dataset

    def __len__(self):
        return len(self.img_classifications)

    def __getitem__(self, idx):
        img_path, classification = self.img_classifications[idx]
        image = Image.open(img_path)
        # Convert to RGB if the image is RBGA, since the model expects RGB images (3 channels)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            classification = self.target_transform(classification)
        return image, classification
    
#Implementaion of early stopping, requires there to be no improvement for 5 rounds to stop the training
class EarlyStopping:
    def __init__(self, patience=5, min_change=0):
        self.patience = patience
        self.min_change = min_change
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_change:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter = self.counter +  1
            if self.counter >= self.patience:
                self.early_stop = True

# Function to Compute APCER and BPCER
def compute_apcer_bpcer(predictions, labels, threshold=0.5):
    # Make a binary prediction based on the threshold
    prediction = (predictions >= threshold).astype(int)
    # Count errors 
    attack_errors = np.sum((prediction == 0) & (labels == 1))
    bona_fide_errors = np.sum((prediction == 1) & (labels == 0))
    # Count total attacks and bona fide
    total_attacks = np.sum(labels == 1)
    total_bona_fide = np.sum(labels == 0)
    # Calculate APCER and BPCER
    apcer = attack_errors / total_attacks if total_attacks > 0 else 0
    bpcer = bona_fide_errors / total_bona_fide if total_bona_fide > 0 else 0
    return apcer, bpcer

def find_optimal_threshold(model, dataloader, device, target_apcer=0.10, precision=0.001):
    # Compute all predictions once
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for image, labels in dataloader:
            image, labels = image.to(device), labels.to(device)
            absolute_prediction = model(image).view(-1).cpu().numpy()
            all_predictions.extend(absolute_prediction)
            all_labels.extend(labels.cpu().numpy())
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Binary search for the optimal threshold
    min_threshold = 0
    max_threshold = 1
    optimal_threshold = None

    while max_threshold - min_threshold > precision:
        # Calculate the mid threshold, which will be the new threshold
        mid_threshold = (min_threshold + max_threshold) / 2
        predicted_labels = (all_predictions >= mid_threshold).astype(int)
        # Calculate the APCER and BPCER for the current threshold with the predicted labels
        apcer, bpcer = compute_apcer_bpcer(predicted_labels, all_labels, mid_threshold)
        accuracy = np.mean(predicted_labels == all_labels)

        print(f"Testing threshold: {mid_threshold:.6f} -> Accuracy: {accuracy:.6f}, APCER: {apcer:.6f}, BPCER: {bpcer:.6f}")
        # Find a new span where the best threshold is located
        if abs(apcer - target_apcer) <= precision:
            return mid_threshold, accuracy, apcer, bpcer
        elif apcer > target_apcer:
            max_threshold = mid_threshold
        else:
            min_threshold = mid_threshold

    optimal_threshold = (min_threshold + max_threshold) / 2
    predicted_labels = (all_predictions >= optimal_threshold).astype(int)
    apcer, bpcer = compute_apcer_bpcer(predicted_labels, all_labels, optimal_threshold)
    accuracy = np.mean(predicted_labels == all_labels)

    return optimal_threshold, accuracy, apcer, bpcer

if __name__ == '__main__':
    # Path for the training images
    training_dataset_path = ""
    # Path for the unseen testing images
    unseen_testing_dataset_path = ""
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Standard transformation to make the dataset more robust
    # Resize the images to 224x224, apply random horizontal flip, random rotation, random affine transformation,
    # color jitter, and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Initialize the dataset and dataloarder
    dataset = CustomImageDataset(training_dataset_path, transform=transform)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    training_dataset, testing_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    dataset_unseen = CustomImageDataset(unseen_testing_dataset_path, transform=transform)
    unseen_test_dataloader = DataLoader(dataset_unseen, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)



    # Training loop
    criterion = nn.BCELoss()
    # Loop for testing multiple learning rates and weight decay values
    for lerningrate in np.linspace(0.01, 0.0001, 3):
        for wd in np.linspace(0.0001, 0.01, 3):
            # Prepare the Early stopping
            early_stopping = EarlyStopping(patience=5, min_change=0.001)
            stopped_epoch=0
            # Load the model
            model = timm.create_model('efficientnet_b1', pretrained=True)
            num_classes = 1
            num_features = model.classifier.in_features
            # Change the head of the model to match the number of classes
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_features, num_classes),
                nn.Sigmoid()  # Sigmoid activation for probabilities
            )
            # Initialize the optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=lerningrate, weight_decay=wd)
            # Initialize the learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

            # Start the training process
            model.to(device)
            best_unseen_accuracy = 0
            for epoch in range(30):
                model.train()
                running_loss = 0.0
                val_loss = 0.0
                # Iterate through the training data
                with torch.no_grad():
                    for image, labels in test_dataloader:
                        # Calculate the validation loss in order to test the early stopping
                        image, labels = image.to(device), labels.to(device)
                        absolute_prediction = model(image)
                        loss = criterion(absolute_prediction, labels.view(-1, 1).float())
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(test_dataloader)
                # Update the learning rate usign ReduceLROnPlateau with the avg_val_loss
                scheduler.step(avg_val_loss)
                unseen_optimal_threshold, unseen_accuracy, unseen_apcer, unseen_bpcer = find_optimal_threshold(model, unseen_test_dataloader, device)
                # save the model if it is the best and has a BPCER10 better than 0.5
                if unseen_accuracy > best_unseen_accuracy:
                    best_unseen_accuracy = unseen_accuracy
                    if unseen_bpcer < 0.5:
                        model_path = f"Trained_epoch{epoch}_total_unseen_accuracy{unseen_accuracy}_APCER{unseen_apcer}_BPCER{unseen_bpcer}_-with-transform_lr{lerningrate}_decay{wd}_{timestr}.pth"
                        torch.save(model.state_dict(), model_path)
                # Update early stopping
                early_stopping(avg_val_loss)
                if early_stopping.early_stop:
                    stopped_epoch = epoch + 1
                    print("Early stopping triggered at epoch:", epoch + 1)
                    break