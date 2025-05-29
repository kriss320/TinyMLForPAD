import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models
import timm
from thop import profile
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchprofile
import os
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

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


def calculate_sparsity(model):
    '''
    Since weights are set to 0 in pytorch instead of removed sparsity is needed 
    Sparcity is an alternative metric to removed weights, but is a count of the weights that are set to 0
    
    Parameters: 
    model: the model which the calculations will be performed on.

    Returns: 
    total_params: the total parameter count of the model
    zero_params: the amount of parameters that are zero
    sparcity: the percentage of parameters that are zero
    '''
    total_params = 0
    zero_params = 0

    # Iterate through all the parameters of the model
    for name, param in model.named_parameters():
        # Count total parameters
        total_params += param.numel()

        # Count zero (pruned) parameters
        zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params if total_params > 0 else 0
    print(f"Total Parameters: {total_params}")
    print(f"Sparsity: {sparsity:.4f}")

    return total_params, zero_params, sparsity
    
#https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
def prune_model(model, prune_amount):
    '''
    Defenition of pruning method. will only prune Conv2d layers and linear layers. 
    Only Conv2d is needed when pruning EfficientNet

    Parameters:
    model: the model the pruning is performed on
    prune_amount: the percentage of weights that are to be removed in the layers

    Returns:
    model: the pruned model
    '''

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):  # Prune Conv2d layers
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            prune.remove(module, 'weight')  # Remove the pruning mask
        elif isinstance(module, torch.nn.Linear):  # Prune Linear layers
            prune.l1_unstructured(module, name='weight', amount=prune_amount)
            prune.remove(module, 'weight')  # Remove the pruning mask
    return model




def freeze_conv_layers(model):
    '''
    Used to freeze the weights of the convolutional layers in the model so they are not changed during finetuning
    Inspired by the pytorch forum post:
    https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/7
    
    Parameters: 
    model: the model which will get frozen layers

    '''
    # Iterate over the model's layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Freeze weights and biases of Conv2D layers
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            print(f"Frozen {name}.weight and {name}.bias")

#not much use for this one when working with EfficientNet
def freeze_fc_layers(model):
    '''
    Used to freeze the weights of the fully connected layers in the model so they are not changed during finetuning
    Inspired by the pytorch forum post:
    https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/7
    
    Parameters: 
    model: the model which will get frozen layers

    '''
    # Freeze fully connected layers (e.g., Linear layers)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            print(f"Frozen {name}.weight and {name}.bias")


def create_optimizer(model, lr=0.01):
    '''
    Create an optimizer for the model, excluding frozen parameters
    Inspired by the pytorch forum post:
    https://discuss.pytorch.org/t/how-to-set-arguments-of-optimizer-when-loading-a-pretrained-net-with-its-gradient-fix-to-a-new-net/18909
    
    Parameters:
    model: the model the optimizer is built for, used to make a filter for which parameters to update
    lr: to set a learning rate to start the scheduler

    Returns:
    optimizer: the optimizer with the frozen weights removed
    '''
    # Filter parameters where requires_grad=True, works in combination with the freezing above
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=1e-5)
    return optimizer

#https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
def fine_tune_model(model, train_dataloader, device, num_epochs=15, lr=0.001, unseen_dataloader=None,prune_amount=0.2, model_path=None):
    '''
    Fine tuning the model after pruning. Uses the same learning methodology as the training.py file

    Parameters:
    model: the model which is finetuned
    train_dataloader: the dataloader with the images used for training
    device: the device the training will be run on
    num_epochs: max amount of epochs for the trainig
    lr: override the default learning rate of 0.001
    unseen_dataloader: the dataloader the model will be tested on to evaluate generalization
    prune_amount: used for loging performance
    model_path: the path where the finetuned model will be saved

    '''
    model.train()
    optimizer = create_optimizer(model, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    criterion = nn.BCELoss()  # Binary cross-entropy 
    best_unseen_accuracy = 0.0
    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evalutaion loss to calculate the learning rate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1).float())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_dataloader)
        scheduler.step(avg_val_loss)

        params, zero_params, sparcity =calculate_sparsity(model)
        unseen_optimal_threshold, unseen_accuracy, unseen_apcer, unseen_bpcer = find_optimal_threshold(model, unseen_dataloader, device)
        # Store the results in a log file, mainly for debuging purposes
        if unseen_accuracy > best_unseen_accuracy and unseen_accuracy > 0.6:
            with open("pruning.log", 'a') as log:
                log.write(f"Pruned  {prune_amount *100:.2f}% of weights tested unseen dataset ,  epochs -> Accuracy: {unseen_accuracy:.6f}, APCER: {unseen_apcer:.6f}, BPCER: {unseen_bpcer:.6f}, params {params}, zero params {zero_params}, sparcity {sparcity}\n")
                log.write("\n")
                log.write("\n")

        # Store the model if the accuracy is better than the previous one and bpcer is lower than 50%
        if unseen_accuracy > best_unseen_accuracy and unseen_bpcer < 0.5:
            model_path_temp = model_path + f"model_name.pth"
            best_unseen_accuracy = unseen_accuracy
            torch.save(model.state_dict(), model_path_temp)
#https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html
if __name__ == '__main__':
    img_file_path = ""
    unseen_img_file_path = ""
    
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
    # Set up the dataset and dataloader
    dataset = CustomImageDataset(img_file_path, transform=transform)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    training_dataset, testing_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    unseen_dataset = CustomImageDataset(unseen_img_file_path, transform=transform)
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)



    for i in range(10):
        # Gradual pruning loop
        for prune_step in range(10, 101 ,5):
            # Load the model
            model = timm.create_model('efficientnet_b0', pretrained=True)
            num_classes = 1
            num_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(num_features, num_classes),
                nn.Sigmoid()  # Sigmoid activation for probabilities
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.load_state_dict(torch.load("model.pth", weights_only=True, map_location=torch.device(device)))
            model.train()


            # Gradually prune 5% more in each step
            prune_amount = prune_step /100
            print(f"\nPruning {prune_amount*100:.2f}% of weights...")
            # Prune the model
            model = prune_model(model, prune_amount)

            # Freeze the layers to not change them after pruning
            freeze_fc_layers(model)
            freeze_conv_layers(model)
            calculate_sparsity(model)
            # Fine-tune the model after pruning
            fine_tune_model(model, train_dataloader, device, num_epochs=15,unseen_dataloader=unseen_dataloader,prune_amount=prune_amount, model_path="Insert_path")
            
            params, zero_params, sparcity = calculate_sparsity(model)
            # Measure performance after fine-tuning
            unseen_treshold, unseen_accuracy, unseen_apcer, unseen_bpcer = find_optimal_threshold(model, unseen_dataloader, device)
            print(f"Optimal unseen Threshold: {unseen_treshold:.6f}")
            print(f"Final Results unseen -> Accuracy: {unseen_accuracy:.6f}, APCER: {unseen_apcer:.6f}, BPCER: {unseen_bpcer:.6f}")

