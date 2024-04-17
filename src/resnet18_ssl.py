import torch
import torchvision.transforms as transforms
from src.data.epic_loader_ssl import EpicDatasetSSL
from torch.utils.data import DataLoader

# Load a pre-trained CNN model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Define a data transformation
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load a dataset (e.g., ImageNet) without labels
dataset = EpicDatasetSSL(data, transform=transform)

# Create a data loader
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Feature extraction
for batch in data_loader:
    images, _ = batch
    with torch.no_grad():
        features = model(images)
    # Features can now be used for downstream tasks