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

# Time for filename consistency
timestr = time.strftime("%Y-%m-%d")

# Dataset Class Definition
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
#https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
#https://github.com/pvgladkov/knowledge-distillation/blob/master/knowledge_distillation/loss.py
#https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/model/net.py?utm_source=chatgpt.com
def distillation_loss(student_outputs, teacher_outputs, labels, alpha=0.5, temperature=3):
    soft_targets = torch.sigmoid(teacher_outputs / temperature)
    kl_loss = F.kl_div(
        torch.log_softmax(student_outputs / temperature, dim=1),
        soft_targets,
        reduction="batchmean"
    ) * (temperature ** 2)

    ground_truth_loss = F.binary_cross_entropy(student_outputs, labels.view(-1, 1).float())
    return alpha * kl_loss + (1 - alpha) * ground_truth_loss

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

# Function to Test the Loaded Model
def test_model(model, dataloader, device, threshold=0.5):
    # Set the model to evaluation mode
    model.eval()
    correct, total = 0, 0
    all_predictions, all_labels = [], []

    with torch.no_grad():
        # Iterate through the DataLoader, and predict the labels
        for batch_idx, (image, labels) in enumerate(dataloader):
            image, labels = image.to(device), labels.to(device)
            absolute_prediction = model(image).view(-1)
            # Threshold-based predictions
            predictions = (absolute_prediction >= threshold).float()

            # Accuracy calculation for the batch
            batch_correct = (predictions == labels.float()).sum().item()
            batch_total = labels.size(0)
            batch_accuracy = batch_correct / batch_total

            # Update overall metrics
            correct += batch_correct
            total += batch_total
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())



    # Calculate overall metrics
    accuracy = correct / total
    apcer, bpcer = compute_apcer_bpcer(np.array(all_predictions), np.array(all_labels), threshold)
    # Print overall metrics
    print(f"Overall Accuracy: {accuracy:.2f}, APCER: {apcer:.2f}, BPCER: {bpcer:.2f}")

    return accuracy, apcer, bpcer
    
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
    # Final threshold
    optimal_threshold = (min_threshold + max_threshold) / 2
    predicted_labels = (all_predictions >= optimal_threshold).astype(int)
    # Calculate the final APCER and BPCER
    apcer, bpcer = compute_apcer_bpcer(predicted_labels, all_labels, optimal_threshold)
    accuracy = np.mean(predicted_labels == all_labels)

    return optimal_threshold, accuracy, apcer, bpcer

def main():
    # Load Training Dataset and Split into Train/Validation
    train_data_path = ""
    full_dataset = CustomImageDataset(train_data_path, transform=transform)

    # Splitting dataset:
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Load Unseen Test Dataset
    test_data_path = ""
    test_dataset = CustomImageDataset(test_data_path, transform=transform)

    # Define Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)#validation dataset on the same dataset as the training dataset
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load Pre-Trained Teacher Model (EfficientNet-B1)
    teacher_model = timm.create_model('efficientnet_b1', pretrained=False)
    teacher_model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 1), nn.Sigmoid())
    teacher_model.load_state_dict(torch.load("models/B1/synt+siw/trained_synthandsiw_timm_efficientnet_b1_stepLR_epoch1_total_unseen_accuracy0.9428571428571428_APCER0.02857142857142857_BPCER0.08571428571428572_-with-transform_lr0.0001_decay0.0001_2025-02-13.pth"))
    teacher_model.to(device)
    teacher_model.eval()  # Freeze the teacher model

    # Define Student Model (EfficientNet-B0)
    student_model = timm.create_model('efficientnet_b0', pretrained=True)
    student_model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 1), nn.Sigmoid())
    student_model.to(device)

    # Training Loop with Unseen Data Testing
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

        avg_train_loss = running_loss / len(train_dataloader)
        # Test the student model on the evaluation dataset
        print("\nStudent Model Threshold Testing:")
        best_threshold, unseen_accuracy, unseen_apcer, unseen_bpcer = find_optimal_threshold(student_model, test_dataloader, device)
        print(f"Epoch {epoch+1}:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Student Model - Optimal Threshold: {best_threshold:.4f}")
        print(f"  Student Model - Accuracy: {unseen_accuracy:.2f}, APCER: {unseen_apcer:.2f}, BPCER: {unseen_bpcer:.2f}")

    # Save Student Model
    torch.save(student_model.state_dict(), f"models/B0/synth+siw/knowledge/student_model_{timestr}.pth")

if __name__ == "__main__":
    main()