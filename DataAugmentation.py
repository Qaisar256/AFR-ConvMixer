import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import resample

# Define a custom Dataset class
class FaceDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx], 'label': self.labels[idx]}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

# Load your data and labels
# training_data, training_labels = ...
# testing_data, testing_labels = ...

# Define transformations for data augmentation
data_augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply data augmentation to training data
augmented_training_data = []
augmented_training_labels = []
for label in set(training_labels):
    label_indices = torch.where(torch.tensor(training_labels) == label)[0]
    minority_class_data = [training_data[i] for i in label_indices]
    augmented_minority_data = []
    for image in minority_class_data:
        augmented_minority_data.append(data_augmentation_transform(image))
    augmented_training_data.extend(augmented_minority_data)
    augmented_training_labels.extend([label] * len(augmented_minority_data))

# Combine original training data with augmented data
augmented_training_data += training_data
augmented_training_labels += training_labels

# Create custom datasets with augmented data
augmented_training_dataset = FaceDataset(augmented_training_data, augmented_training_labels)
testing_dataset = FaceDataset(testing_data, testing_labels, transform=data_augmentation_transform)

# Create DataLoader instances for training and testing
batch_size = 32
train_loader = DataLoader(augmented_training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

# Now use train_loader and test_loader for training and testing
