import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

import sys

class DynamicLogger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
        self.filename = filename

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def change_log_file(self, new_filename):
        """Change to a new log file."""
        self.flush()
        self.log.close()
        self.filename = new_filename
        self.log = open(new_filename, "a", encoding='utf-8')

logger = DynamicLogger("initial_log.txt")
sys.stdout = logger

#-----------------------------------------------------

dataset_dict = {
    'paultimothymooney':r'C:\Users\TimePC\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray',
    'mimic_cxr':r'C:\Users\TimePC\.cache\kagglehub\datasets\itsanmol124\mimic-cxr\versions\1',
    'chexpert':r'C:\Users\TimePC\.cache\kagglehub\datasets\mimsadiislam\chexpert\versions\1',
    'chestx_ray14':r'C:\Users\TimePC\.cache\kagglehub\datasets\lawhan\chestx-ray14\versions\1\ChestX-ray14',
    }

# Define dataset paths
for dataset_name,data_dir in dataset_dict.items():
    for transform_mode in ["scale", "zscore", "normalize"]:
        for model_name in ["cnn", "mobilenetv2", "efficientnet_b0"]:
            save_path = 'results'

            input_channels = 1  # Grayscale images
            img_height = 256
            img_width = 256

            num_classes = 2  # Adjust as needed

            batch_size = 32
            epochs = 50
            patience = 50  # or any number of epochs you want to wait
            num_workers = 0 # 0 for Windows compatibility

            train_size = 1000
            val_size = 200
            # test_size = 1000

            #-----------------------------------------------------

            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            run_name = f"{dataset_name}_{transform_mode}_{model_name}"
            log_filename = os.path.join(save_path,f"log_{run_name}.txt")
            model_path = os.path.join(save_path,f"model_{run_name}.pth")
            image_path = os.path.join(save_path,f"metric_{run_name}.png")

            os.makedirs(save_path,exist_ok=True)

            print('data_dir :',data_dir)
            print('image (H,W,C) :',img_height,'x',img_width,'x',input_channels)
            print('datadataset_name_dir :',dataset_name)
            print('transform_mode :',transform_mode)
            print('model_name :',model_name)
            print('num_classes :',num_classes)
            print('batch_size :',batch_size)
            print('epochs :',epochs)
            print('num_workers :',num_workers)

            #-----------------------------------------------------


            # Set up logger
            logger.change_log_file(log_filename)

            def set_seed(seed=42):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            set_seed(42)  # You can change the seed number

            #-----------------------------------------------------
            if transform_mode in ["zscore", "normalize"]:

                # Function to calculate mean and std
                def calculate_mean_std(loader):
                    mean = 0.0
                    std = 0.0
                    total_images_count = 0
                    for images, _ in loader:
                        images = images.view(images.size(0), images.size(1), -1)
                        mean += images.mean(2).sum(0)
                        std += images.std(2).sum(0)
                        total_images_count += images.size(0)
                    mean /= total_images_count
                    std /= total_images_count
                    return mean, std

                # Load datasets 
                # ['train', 'val', 'test']

                # Define transformations
                calculate_mean_std_transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((img_height, img_width)),
                    transforms.ToTensor(),
                ])

                # Load the full train dataset
                full_train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=calculate_mean_std_transform)

                # Randomly sample 10,000 indices (or all if less than 10,000)
                num_samples = min(train_size, len(full_train_dataset))
                sample_indices = np.random.choice(len(full_train_dataset), num_samples, replace=False)
                sampled_train_dataset = Subset(full_train_dataset, sample_indices)

                datasets = {'train': sampled_train_dataset}
                loaders = {'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers)}

                # Calculate mean and std for each dataset
                mean_std = {x: calculate_mean_std(loaders[x]) for x in ['train']}
                target_mean = mean_std['train'][0].numpy()[0]
                target_std = mean_std['train'][1].numpy()[0]
                print(target_mean,target_std)

                # Define transformations
                transform_zscore = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((img_height, img_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[target_mean], std=[target_std])
                ])

            #-----------------------------------------------------

            if transform_mode == "scale":
                # Define transformations
                transform_scale = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((img_height, img_width)),
                    transforms.ToTensor()
                ])

                # Load datasets
                train_dataset_full = ImageFolder(os.path.join(data_dir, 'train'), transform=transform_scale)
                # val_dataset_full   = ImageFolder(os.path.join(data_dir, 'val'), transform=transform_scale)
                # test_dataset_full  = ImageFolder(os.path.join(data_dir, 'test'), transform=transform_scale)
                val_dataset_full  = ImageFolder(os.path.join(data_dir, 'test'), transform=transform_scale)

                # Sample 8,000 images from the training set (or all if less)
                num_samples = min(train_size, len(train_dataset_full))
                sample_indices = np.random.choice(len(train_dataset_full), num_samples, replace=False)
                train_dataset = Subset(train_dataset_full, sample_indices)
                # Sample 2,000 images from the validation set (or all if less)
                num_samples = min(val_size, len(val_dataset_full))
                sample_indices = np.random.choice(len(val_dataset_full), num_samples, replace=False)
                val_dataset = Subset(val_dataset_full, sample_indices)
                # Sample 1,000 images from the testing set (or all if less)
                # num_samples = min(1000, len(test_dataset_full))
                # sample_indices = np.random.choice(len(test_dataset_full), num_samples, replace=False)
                # test_dataset = Subset(test_dataset_full, sample_indices)

                # Data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                # test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            elif transform_mode == "zscore":
                # Load datasets
                train_dataset_full = ImageFolder(os.path.join(data_dir, 'train'), transform=transform_zscore)
                # val_dataset_full   = ImageFolder(os.path.join(data_dir, 'val'), transform=transform_zscore)
                # test_dataset_full  = ImageFolder(os.path.join(data_dir, 'test'), transform=transform_zscore)
                val_dataset_full  = ImageFolder(os.path.join(data_dir, 'test'), transform=transform_zscore)

                # Sample 8,000 images from the training set (or all if less)
                num_samples = min(train_size, len(train_dataset_full))
                sample_indices = np.random.choice(len(train_dataset_full), num_samples, replace=False)
                train_dataset = Subset(train_dataset_full, sample_indices)
                # Sample 2,000 images from the validation set (or all if less)
                num_samples = min(val_size, len(val_dataset_full))
                sample_indices = np.random.choice(len(val_dataset_full), num_samples, replace=False)
                val_dataset = Subset(val_dataset_full, sample_indices)
                # Sample 1,000 images from the testing set (or all if less)
                # num_samples = min(1000, len(test_dataset_full))
                # sample_indices = np.random.choice(len(test_dataset_full), num_samples, replace=False)
                # test_dataset = Subset(test_dataset_full, sample_indices)

                # Data loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                # test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                
            elif transform_mode == "normalize":

                class CustomDataset(Dataset):
                    def __init__(self, image_dir, cache_dir, img_height, img_width, target_mean, target_std, transform=None):
                        self.image_dir = image_dir
                        self.transform = transform
                        self.cache_dir = cache_dir
                        self.img_height = img_height
                        self.img_width = img_width
                        self.target_mean = target_mean
                        self.target_std = target_std
                        os.makedirs(cache_dir, exist_ok=True)
                        # Cache processed images and labels
                        self._cache_data()

                        self.images = []
                        self.labels = []
                        self.classes = os.listdir(self.cache_dir)
                        for class_idx, class_name in enumerate(self.classes):
                            class_dir = os.path.join(self.cache_dir, class_name)
                            for img_name in os.listdir(class_dir):
                                img_path = os.path.join(class_dir, img_name)
                                self.images.append(img_path)
                                self.labels.append(class_idx)
                        
                    def _cache_data(self):
                        """Caches preprocessed images and labels as tensors."""
                        self.classes = os.listdir(self.image_dir)
                        for class_idx, class_name in enumerate(self.classes):
                            class_dir = os.path.join(self.image_dir, class_name)
                            for img_name in os.listdir(class_dir):
                                img_path = os.path.join(class_dir, img_name)
                                    
                                cache_class_path = os.path.join(self.cache_dir,class_name)
                                os.makedirs(cache_class_path, exist_ok=True)

                                cache_img_path = os.path.join(cache_class_path, img_name)
                                
                                if not os.path.exists(cache_img_path):
                                            
                                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                                    # Compute sum of grayscale values along x-axis (rows)
                                    gray_sum_rows = np.sum(img, axis=1)
                                    
                                    # Compute sum of grayscale values along y-axis (columns)
                                    gray_sum_columns = np.sum(img, axis=0)
                                    
                                    # Calculate CDF of gray_sum_rows
                                    gray_sum_rows_cdf = np.cumsum(gray_sum_rows) / np.sum(gray_sum_rows)
                                    
                                    # Calculate CDF of gray_sum_columns
                                    gray_sum_columns_cdf = np.cumsum(gray_sum_columns) / np.sum(gray_sum_columns)
                                    
                                    # Find the indices for 0.1 and 0.9 CDF for x and y
                                    x_min = np.searchsorted(gray_sum_columns_cdf, 0.05)
                                    x_max = np.searchsorted(gray_sum_columns_cdf, 0.95)
                                    y_min = np.searchsorted(gray_sum_rows_cdf, 0.15)
                                    y_max = np.searchsorted(gray_sum_rows_cdf, 0.95)
                                    
                                    # Crop the image
                                    img_crop = img[ y_min:y_max,x_min:x_max]
                                    img_crop = cv2.resize(img_crop, (self.img_height,self.img_width))
                                        
                                    # Compute mean and standard deviation
                                    mean_gray = np.mean(img_crop)
                                    std_gray = np.std(img_crop)
                                    
                                    # Normalize image
                                    norm_img = (img_crop - mean_gray) / std_gray  # Standardize
                                    norm_img = (norm_img * self.target_std*255) + self.target_mean*255  # Scale to new distribution
                                    
                                    # Clip values to valid range
                                    norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)
                                    cv2.imwrite(cache_img_path,norm_img)

                    def __len__(self):
                        return len(self.images)
                    
                    def __getitem__(self, idx):
                        cache_img_path = self.images[idx]
                        label = self.labels[idx]
                        
                        img_pil = Image.open(cache_img_path)

                        if self.transform:
                            image = self.transform(img_pil)
                        
                        return image, label

                # Create custom dataset
                train_dataset_full = CustomDataset(
                    image_dir=os.path.join(data_dir, "train"), 
                    cache_dir=os.path.join(data_dir, "train_cache"), 
                    img_width=img_width,
                    img_height=img_height,
                    target_mean=target_mean, 
                    target_std=target_std, 
                    transform=transform_zscore)
                # val_dataset_full = CustomDataset(
                #     image_dir=os.path.join(data_dir, "val"), 
                #     cache_dir=os.path.join(data_dir, "val_cache"), 
                #     img_width=img_width,
                #     img_height=img_height,
                #     target_mean=target_mean, 
                #     target_std=target_std, 
                #     transform=transform_zscore)
                # test_dataset_full = CustomDataset(
                #     image_dir=os.path.join(data_dir, "test"), 
                #     cache_dir=os.path.join(data_dir, "test_cache"), 
                #     img_width=img_width,
                #     img_height=img_height,
                #     target_mean=target_mean, 
                #     target_std=target_std, 
                #     transform=transform_zscore)
                val_dataset_full = CustomDataset(
                    image_dir=os.path.join(data_dir, "test"), 
                    cache_dir=os.path.join(data_dir, "test_cache"), 
                    img_width=img_width,
                    img_height=img_height,
                    target_mean=target_mean, 
                    target_std=target_std, 
                    transform=transform_zscore)

                # Sample 8,000 images from the training set (or all if less)
                num_samples = min(train_size, len(train_dataset_full))
                sample_indices = np.random.choice(len(train_dataset_full), num_samples, replace=False)
                train_dataset = Subset(train_dataset_full, sample_indices)
                # Sample 2,000 images from the validation set (or all if less)
                num_samples = min(val_size, len(val_dataset_full))
                sample_indices = np.random.choice(len(val_dataset_full), num_samples, replace=False)
                val_dataset = Subset(val_dataset_full, sample_indices)
                # Sample 1,000 images from the testing set (or all if less)
                # num_samples = min(1000, len(test_dataset_full))
                # sample_indices = np.random.choice(len(test_dataset_full), num_samples, replace=False)
                # test_dataset = Subset(test_dataset_full, sample_indices)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                # test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                # ...existing code...
            #-----------------------------------------------------

            # Class names
            train_classes = train_dataset_full.classes
            print("Class Names:", train_classes)
            print("train size:", len(train_dataset))
            print("val size:", len(val_dataset))

            #-----------------------------------------------------

            # ---------------------------
            # 1. Original CNN Model
            # ---------------------------
            class CNN(nn.Module):
                def __init__(self, num_classes):
                    super(CNN, self).__init__()
                    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                    self.fc1 = nn.Linear(128 * (img_height // 8) * (img_width // 8), 128)
                    self.fc2 = nn.Linear(128, num_classes)
                    self.relu = nn.ReLU()
                    self.softmax = nn.Softmax(dim=1)
                
                def forward(self, x):
                    x = self.pool(self.relu(self.conv1(x)))
                    x = self.pool(self.relu(self.conv2(x)))
                    x = self.pool(self.relu(self.conv3(x)))
                    x = x.view(x.size(0), -1)
                    x = self.relu(self.fc1(x))
                    x = self.softmax(self.fc2(x))
                    return x

            # ---------------------------
            # 2. MobileNetV2
            # ---------------------------
            def build_mobilenetv2(num_classes):
                model = models.mobilenet_v2(pretrained=True)

                # Extract original Conv2d layer from Conv2dNormActivation
                old_layer = model.features[0]
                old_conv = old_layer[0]  # This is the actual nn.Conv2d layer

                # Create new Conv2d with 1 input channel, keeping all other parameters the same
                new_conv = nn.Conv2d(
                    in_channels=1,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )

                # Replace the Conv2d in Conv2dNormActivation
                model.features[0][0] = new_conv

                # Adjust classifier
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

                return model

            # ---------------------------
            # 3. EfficientNet-B0
            # ---------------------------
            def build_efficientnet_b0(num_classes):
                model = models.efficientnet_b0(pretrained=True)
                # Adapt input conv to 1 channel
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                return model

            # ---------------------------
            # Model Selection
            # ---------------------------

            if model_name == "cnn":
                model = CNN(num_classes)
            elif model_name == "mobilenetv2":
                model = build_mobilenetv2(num_classes)
            elif model_name == "efficientnet_b0":
                model = build_efficientnet_b0(num_classes)
            else:
                raise ValueError("Invalid model name")

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

            # Learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            #-----------------------------------------------------

            print('Start training')
            # Train the model
            train_losses, train_accs = [], []
            val_losses, val_accs = [], []
            # test_losses, test_accs = [], []
            
            train_y_true, train_y_pred, train_y_scores = [], [], []
            val_y_true, val_y_pred, val_y_scores = [], [], []
            # test_y_true, test_y_pred, test_y_scores = [], [], []

            for epoch in range(epochs):
                model.train()
                train_running_loss, train_correct, train_total = 0.0, 0, 0

                optimizer.zero_grad()
                
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_running_loss += loss.item()
                
                    train_y_true.extend(labels.cpu().numpy())
                    train_y_pred.extend(predicted.cpu().numpy())
                    train_y_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu()[:, 1])

                optimizer.step()

                train_loss = train_running_loss / len(train_loader)
                train_acc = train_correct / train_total
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                scheduler.step()
                
                model.eval()
                val_running_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        val_running_loss += loss.item()
                
                        val_y_true.extend(labels.cpu().numpy())
                        val_y_pred.extend(predicted.cpu().numpy())
                        val_y_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu()[:, 1])

                val_loss = val_running_loss / len(val_loader)
                val_acc = val_correct / val_total
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()  # Save best model
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        model.load_state_dict(best_model_state)  # Restore best model
                        break

            # test_running_loss, test_correct, test_total = 0.0, 0, 0
            # with torch.no_grad():
            #     for images, labels in test_loader:
            #         images, labels = images.to(device), labels.to(device)
            #         outputs = model(images)
            #         loss = criterion(outputs, labels)
                    
            #         _, predicted = torch.max(outputs, 1)
            #         test_total += labels.size(0)
            #         test_correct += (predicted == labels).sum().item()
            #         test_running_loss += loss.item()

            #         test_y_true.extend(labels.cpu().numpy())
            #         test_y_pred.extend(predicted.cpu().numpy())
            #         test_y_scores.extend(torch.nn.functional.softmax(outputs, dim=1).cpu()[:, 1])
            # test_loss = test_running_loss / len(test_loader)
            # test_acc = test_correct / test_total
            # test_losses.append(test_loss)
            # test_accs.append(test_acc)
            # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            #-----------------------------------------------------

            # Classification report
            train_report = classification_report(train_y_true, train_y_pred, target_names=train_classes, output_dict=True)
            train_df_report = pd.DataFrame(train_report).transpose()
            print('train_report')
            print(train_df_report)

            val_report = classification_report(val_y_true, val_y_pred, target_names=train_classes, output_dict=True)
            val_df_report = pd.DataFrame(val_report).transpose()
            print('val_report')
            print(val_df_report)

            # test_report = classification_report(test_y_true, test_y_pred, target_names=train_classes, output_dict=True)
            # test_df_report = pd.DataFrame(test_report).transpose()
            # print('test_report')
            # print(test_df_report)

            #-----------------------------------------------------


            # Plot metrics
            plt.figure(figsize=(12, 6))

            plt.subplot(2, 2, 1)
            plt.plot(train_accs, label='Train Accuracy')
            plt.plot(val_accs, label='Validation Accuracy')
            plt.ylim(0, 1)
            plt.legend()
            plt.title("Model Accuracy")

            plt.subplot(2, 2, 2)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
            plt.title("Model Loss")

            plt.subplot(2, 2, 3)
            plt.bar(val_df_report.index[:-1], val_df_report['precision'][:-1], color='blue', alpha=0.7, label='Precision')
            plt.bar(val_df_report.index[:-1], val_df_report['recall'][:-1], color='red', alpha=0.5, label='Recall')
            plt.bar(val_df_report.index[:-1], val_df_report['f1-score'][:-1], color='green', alpha=0.5, label='F1-score')
            plt.xticks(rotation=45)
            plt.legend()
            plt.title("Precision, Recall, and F1-score")

            plt.subplot(2, 2, 4)
            fpr, tpr, _ = roc_curve(val_y_true, val_y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.title("ROC Curve")

            # Save the figure
            plt.tight_layout()
            plt.savefig(image_path, dpi=300)
            # plt.show()

            #-----------------------------------------------------

            # Save the model
            torch.save(model.state_dict(), model_path)
            print("Model saved as xray_classifier.pth")
            logger.flush()