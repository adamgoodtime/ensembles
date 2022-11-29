import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import transforms, datasets

gpu = True
# with torch.no_grad():
# torch.set_grad_enabled(False)
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))


input_size = 784
hidden_size = [128, 64]
output_size = 10


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()

        # Inputs to hidden layer linear transformation
        self.layer1 = nn.Linear(input_size, hidden_size[0])
        # Hidden layer 1 to HL2 linear transformation
        self.layer2 = nn.Linear(hidden_size[0], hidden_size[1])
        # HL2 to output linear transformation
        self.layer3 = nn.Linear(hidden_size[1], output_size)

        # Define relu activation and LogSoftmax output
        self.relu = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # HL1 with relu activation
        out = self.relu(self.layer1(x))
        # HL2 with relu activation
        out = self.relu(self.layer2(out))
        # Output layer with LogSoftmax activation
        out = self.LogSoftmax(self.layer3(out))
        return out


model = NeuralNet(input_size, hidden_size, output_size).to(torch.device("cuda"))

lossFunction = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.9)



num_epochs = 20
for epoch in range(num_epochs):
    loss_ = 0
    for images, labels in train_loader:
        # Flatten the input images of [28,28] to [1,784]
        images = images.reshape(-1, 784).to(torch.device("cuda"))

        # Forward Pass
        output = model(images)
        # Loss at each oteration by comparing to target(label)
        loss = lossFunction(output, labels)

        # Backpropogating gradient of loss
        optimizer.zero_grad()
        loss.backward()

        # Updating parameters(weights and bias)
        optimizer.step()

        loss_ += loss.item()
    print("Epoch{}, Training loss:{}".format(epoch, loss_ / len(train_loader)))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 784).to(torch.device("cuda"))
            out = model(images)
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Testing accuracy: {} %'.format(100 * correct / total))

# torch.save(model, 'mnist_model.pt')

print('done')
