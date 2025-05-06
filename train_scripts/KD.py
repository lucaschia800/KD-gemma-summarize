import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from datasets import load_dataset, concatenate_datasets, load_from_disk


xsum_train = load_from_disk("mistral-KD/data/xsum_formatted")
cnn_train = load_from_disk("mistral-KD/data/cnn_formatted")
sci1_train = load_from_disk("mistral-KD/data/sci1_formatted")

train_ds = concatenate_datasets([xsum_train, cnn_train, sci1_train])


train_ds.set_format(type='torch')


def train(teacher, student, train_ds, device, num_epochs=1, batch_size=32, T = 1.0):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        student.train()
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)

            # Compute loss

            soft_targets = nn.functional.softmax(teacher_outputs / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_outputs / T, dim=-1)


            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            soft_label_loss = criterion(student_outputs, teacher_outputs)
            hard_label_loss = criterion(student_outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
