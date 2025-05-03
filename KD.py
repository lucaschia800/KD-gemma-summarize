import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk


xsum_train = load_from_disk("mistral_KD/data/xsum_formatted")
cnn_train = load_from_disk("mistral_KD/data/cnn_formatted")
sci1_train = load_from_disk("mistral_KD/data/sci1_formatted")

train_ds = concatenate_datasets([xsum_train, cnn_train, sci1_train])


train_ds.set_format(type='torch')


def train(teacher, student, train_ds, device, num_epochs=1, batch_size=32):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    # DataLoader for training data
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        student.train()
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass through teacher and student models
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)

            # Compute loss
            loss = criterion(student_outputs, teacher_outputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
