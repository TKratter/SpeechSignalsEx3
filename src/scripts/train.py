from src.datasets.wav_dataset import WavDataset, char_map
from src.models.encoders import MFCCEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda"

batch_size = 256

num_classes = len(char_map)

model = MFCCEncoder(input_dim=20, hidden_dim=256, output_dim=num_classes, feature_extractor_depth=3, lstm_layers=2).to(
    device)
optimizer = optim.Adam(model.parameters())

train_dataset = WavDataset(path='/home/tomk42/PycharmProjects/SpeechSignalsEx3/train', device=device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CTCLoss()

num_epochs = 10
print_every = 10

# Training loop
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):

        (audio, labels, input_lengths, target_lengths) = sample.values()

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(audio, input_lengths)

        # Compute the loss
        if batch_size == 1:
            output_lengths = torch.tensor(output.shape[0], dtype=torch.int32)
        else:
            output_lengths = torch.full((output.shape[0],), output.shape[1], dtype=torch.int32)
        if batch_size > 1:
            output = output.permute(1, 0, 2)
        loss = criterion(output, labels, output_lengths, target_lengths)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print the current loss
        if i % print_every == 0:
            print(f'Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'asr_model.pth')
