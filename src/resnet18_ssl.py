'''
Codebase from https://medium.com/@zhonghong9998/exploring-self-supervised-learning-training-without-labeled-data-6e1a47dc5876
'''

import torch
import torchvision.transforms as transforms
from data.epic_dataset_ssl import EpicDatasetSSL
from torch.utils.data import DataLoader
import os

#TODO: Implement resnet feature extraction
#TODO: Implement the SSL pipeline

root_dir = os.path.join('/data', 'EPIC-KITCHENS')
annotations_dir = os.path.join('data', 'annotations')
train = True
filename_training = 'EPIC_100_train_clean.pkl'
def dataloader(root_dir, annotations_dir, filename, train=True):
    data = EpicDatasetSSL(
        src_dir=root_dir,
        annotations=annotations_dir,
        filename=filename
        )

    # Create a data loader
    loader = DataLoader(data, batch_size=32, shuffle=train)

    return loader

def pretrain(model, train_loader, num_epochs=10):
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            images, labels = batch
            images.cuda()
            labels.cuda()

            # Forward pass
            outputs = model(images)
            prova = None


if __name__ == '__main__':
    train_loader = dataloader(root_dir, annotations_dir, filename_training)
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    model.eval()
    model.cuda()

    pretrain(model, train_loader, num_epochs=10)
