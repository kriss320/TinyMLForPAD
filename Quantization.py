import torch
import timm
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Dataset Class Definition
# Defines a dataset which is used in all code-files. 
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


def get_model_size(model, filename="temp.pth"):
    '''
    Function to Get Model Size. Works by storing the model, checking the size of it and removing it
    
    Parameters:
    model: model to get the size of
    filename: temporary filename for the stored model, default temp.pth

    Returns:
    size: a numerical representation of the size of the model
    '''


    torch.save(model.state_dict(), filename)
    size = os.path.getsize(filename)
    os.remove(filename)  # Clean up temporary file
    return size

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


#https://docs.pytorch.org/docs/stable/quantization.html
if __name__ == '__main__':
    # Load the trained model 
    model = timm.create_model('efficientnet_b1', pretrained=True)
    num_classes = 1  
    num_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(num_features, num_classes),
        torch.nn.Sigmoid()
    )
    model.load_state_dict(torch.load("model.pth"))

    # Get original model size
    original_size = get_model_size(model)

    # Apply static quantization
    backend = "qnnpack"
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_dynamic_quantized = torch.quantization.prepare(model, inplace=False)
    model_dynamic_quantized = torch.quantization.convert(model_dynamic_quantized, inplace=False)

    # Get quantized model size
    quantized_size = get_model_size(model_dynamic_quantized)

    # Print size difference
    size_diff = original_size - quantized_size
    print(f"\nOriginal Model Size: {original_size} KB")
    print(f"Quantized Model Size: {quantized_size} KB")
    print(f"Absolute Size Reduction: {size_diff} KB")

    #Define image transformation to match model's input(224x224) and make the dataset more varied
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Set up the dataset and dataloader for testing
    test_img_path = ""  # Path to test images
    test_dataset = CustomImageDataset(img_file_path=test_img_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #Test the model with the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Find the threshold where the APCER is 10%, this will give the classification accuracy, APCER, and BPCER
    target_apcer = 0.10  # Target APCER of 10%
    optimal_threshold, accuracy, apcer, bpcer = find_optimal_threshold(model, test_dataloader, device, target_apcer=target_apcer)
    # Print the final results
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}, APCER: {apcer:.4f}, BPCER: {bpcer:.4f}")
