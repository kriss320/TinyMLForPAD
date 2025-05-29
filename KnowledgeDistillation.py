import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import timm
import numpy as np
import time
from PIL import Image

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Time used in filename
timestr = time.strftime("%Y-%m-%d")

# Dataset Class Definition
# Defines a dataset which is used in all code-files. 
#[1]
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
        np.random.shuffle(self.img_classifications) # shuffle the dataset

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

# Data Transforms
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Knowledge Distillation Loss, inspired by:
#[6][7][8]
def distillation_loss(student_outputs, teacher_outputs, labels ):
    '''
    Defenition of the distillation loss. 
    will give a loss factor for how good the student model is to predict the teachers output
    
    Parameters:
    student_outputs: the numerical prediction from the student model
    teacher_outputs: the numerical predictions from a teacher model
    labels: the correct labels

    returns:
    the distillation loss which is a prediction on the performance of the student model
    '''

    soft_targets = torch.sigmoid(teacher_outputs / 3)
    kl_loss = F.kl_div(
        torch.log_softmax(student_outputs / 3, dim=1),
        soft_targets,
        reduction="batchmean"
    ) * (3 ** 2)

    ground_truth_loss = F.binary_cross_entropy(student_outputs, labels.view(-1, 1).float())
    return 0.5 * kl_loss + (1 - 0.5) * ground_truth_loss

def compute_apcer_bpcer(predictions, labels, threshold=0.5):
    '''
    Function to compute the APCER and BPCER with a threshold.
    Is used by other functions such as find_optimal_threshold

    Parameters
    Predictions: np.array of the predictions the model has made
    labels: np.array of the actuall labels of the images
    threshold= the threshold where images are classified as Bona fide or PAs

    returns:
    apcer: the apcer at the set threshold
    bpcer: the bpcer at the set threshold
    '''
    # Make a binary prediction based on the threshold
    prediction = (predictions >= threshold).astype(int)
    # Count errors 
    attack_errors = np.sum((prediction == 0) & (labels == 1))
    bona_fide_errors = np.sum((prediction == 1) & (labels == 0))
    # Count total attacks and total bona fide
    total_attacks = np.sum(labels == 1)
    total_bona_fide = np.sum(labels == 0)
    # Calculate APCER and BPCER
    apcer = attack_errors / total_attacks if total_attacks > 0 else 0
    bpcer = bona_fide_errors / total_bona_fide if total_bona_fide > 0 else 0
    return apcer, bpcer


def find_optimal_threshold(model, dataloader, device, target_apcer=0.10, precision=0.001):
    '''
    Function to Test the Loaded Model to find the optimal threshold
    Has implemented caching to reduce time spent trying to find the optimal threshold
    uses a simple binary search method to find the optimal threshold, could be optimized using a smarter search

    Parameters:
    model: the model which will be used
    dataloader: the dataloader with the images for testing
    device: the device the model and dataloader will be run on
    target_apcer: the apcer which the threshold will be changed to reach
    precision: the precision needed in the apcer before stoping

    returns:
    optimal_threshold: the threshold where the apcer is met
    accuracy: the total classification accuracy at the threshold
    apcer: the apcer at the set threshold
    bpcer: the bpcer at the set threshold
    '''
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
    optimal_metrics = None

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
    
[9]
def main():
    # Load Training Dataset and Split into Train/Validation
    train_data_path = ""
    train_dataset = CustomImageDataset(train_data_path, transform=transform)

    # Load Unseen Test Dataset
    test_data_path = ""
    test_dataset = CustomImageDataset(test_data_path, transform=transform)

    # Define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load pre-trained teacher model
    teacher_model = timm.create_model('efficientnet_b1', pretrained=False)
    teacher_model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 1), nn.Sigmoid())
    teacher_model.load_state_dict(torch.load("model.pth"))
    teacher_model.to(device)
    teacher_model.eval()  # freeze the teacher model

    # Define student model 
    student_model = timm.create_model('efficientnet_b0', pretrained=True)
    student_model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 1), nn.Sigmoid())
    student_model.to(device)

    # Training loop with unseen data testing
    criterion = distillation_loss
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, weight_decay=0.0001)
    #15 epochs trainig loop
    for epoch in range(15):
        student_model.train()
        running_loss = 0.0
        # Training the student model to mimic the teacher model
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            student_outputs = student_model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            loss = criterion(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        best_threshold, unseen_accuracy, unseen_apcer, unseen_bpcer = find_optimal_threshold(student_model, test_dataloader, device)
        
        if unseen_bpcer < 0.5:
            torch.save(student_model.state_dict(), f"student_model_{timestr}_epoch{epoch}_apcer{unseen_apcer}_bpcer{unseen_bpcer}.pth")
        


        # Test the student model on the evaluation dataset
        print("\nStudent Model Threshold Testing:")
        best_threshold, unseen_accuracy, unseen_apcer, unseen_bpcer = find_optimal_threshold(student_model, test_dataloader, device)
        print(f"Epoch {epoch+1}:")
        print(f"  Student Model - Optimal Threshold: {best_threshold:.4f}")
        print(f"  Student Model - Accuracy: {unseen_accuracy:.2f}, APCER: {unseen_apcer:.2f}, BPCER: {unseen_bpcer:.2f}")

if __name__ == "__main__":
    main()
