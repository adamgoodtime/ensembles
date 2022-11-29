import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import transforms, datasets

torch.manual_seed(27)
np.random.seed(27)

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

class net(nn.Module):
    def __init__(self, input_size, layers, output_size, other_indexes):
        super(net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conn = nn.ModuleList()
        self.other_indexes = other_indexes

        self.add_layers(layers)
        self.activation_list = []

        self.relu = nn.ReLU()
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x, other_act):
        x = torch.hstack([x, other_act[:, self.other_indexes]])
        self.activation_list = []
        for layer in self.conn[:-1]:
            x = self.relu(layer(x))
            self.activation_list.append(x)
        self.activation_list.append(self.logsoft(self.conn[-1](x)))
        return self.activation_list[-1]

    def freeze_all_network(self):
        for para in self.parameters():
            para.requires_grad = False

    def add_layers(self, layers, connect_back=False):
        l = np.hstack([input_size+len(self.other_indexes), layers, output_size])
        layers = [[l[i], l[i+1]] for i in range(len(l)-1)]
        for pre, post in layers:
            self.conn.append(nn.Linear(pre, post))

    def return_act(self):
        return torch.hstack(self.activation_list)


batch_size = 64
trainset = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
testset = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                           generator=torch.Generator(device='cuda'))
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,
                                          generator=torch.Generator(device='cuda'))

input_size = 784
hidden_size = [128, 64]
output_size = 10

models = nn.ModuleList()
num_models = 1
project_amount = 100
all_act = False
training_coupling = 0

learning_rate = 0.01
num_epochs = 2000
model_back_projections = [[]]
model_fixed_activations = []

settings_label = "e-{} lr-{} pa-{} tc-{} hs-{}".format(
    num_epochs, learning_rate, project_amount, training_coupling, hidden_size)
print("Settings:", settings_label)

lossFunction = nn.NLLLoss()

all_training_losses = []
all_testing_accuracy = []

for current_network in range(num_models):
    training_losses = []
    testing_accuracy = []
    print("\nStarting network", current_network)
    if all_act:
        possible_back_connections = current_network*(np.sum(hidden_size)+output_size)
    else:
        possible_back_connections = current_network * output_size
    if project_amount < possible_back_connections:
        projections_back = np.random.choice(range(0, possible_back_connections),
                                            project_amount,
                                            replace=False)
    else:
        projections_back = range(0, possible_back_connections)
    models.append(net(input_size, hidden_size, output_size, projections_back).to(torch.device("cuda")))
    optimizer = optim.SGD(models[-1].parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        loss_ = 0
        for images, labels in train_loader:
            # Flatten the input images of [28,28] to [1,784]
            images = images.reshape(-1, 784).to(torch.device("cuda"))

            # Forward Pass
            if training_coupling or project_amount > 0:
                summed_output = torch.zeros([images.shape[0], output_size]).to(torch.device("cuda"))
                other_act = [torch.zeros([images.shape[0], 0]).to(torch.device("cuda"))]
                for m in models:
                    output = m.forward(images, torch.hstack(other_act))
                    summed_output += output
                    if all_act:
                        other_act.append(m.return_act())
                    else:
                        other_act.append(output)
                if not training_coupling:
                    summed_output = output
            else:
                other_act = [torch.zeros([images.shape[0], 0]).to(torch.device("cuda"))]
                summed_output = models[-1].forward(images, torch.hstack(other_act))
                # save the activations for future runs here
            # Loss at each oteration by comparing to target(label)
            loss = lossFunction(summed_output, labels)

            # Backpropogating gradient of loss
            optimizer.zero_grad()
            loss.backward()

            # Updating parameters(weights and bias)
            optimizer.step()

            loss_ += loss.item()
        print("Net{} Epoch{}, Training loss:{}".format(current_network, epoch, loss_ / len(train_loader)))
        training_losses.append(loss_ / len(train_loader))

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 784).to(torch.device("cuda"))
                summed_output = torch.zeros([images.shape[0], output_size]).to(torch.device("cuda"))
                other_act = [torch.zeros([images.shape[0], 0]).to(torch.device("cuda"))]
                for m in models:
                    output = m.forward(images, torch.hstack(other_act))
                    summed_output += output
                    if all_act:
                        other_act.append(m.return_act())
                    else:
                        other_act.append(output)
                _, predicted = torch.max(summed_output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Testing accuracy: {} %'.format(100 * correct / total))
            testing_accuracy.append(100 * correct / total)
            print("For settings:", settings_label)
            print("Training losses:", training_losses)
            print("Testing accuracies:", testing_accuracy)
    all_training_losses.append(training_losses)
    all_testing_accuracy.append(testing_accuracy)
    print("All training losses:")
    for l in all_training_losses:
        print(l)
    print("All testing accuracies:")
    for a in all_testing_accuracy:
        print(a)
    print(current_network, "networks with settings:", settings_label)
    for m in models:
        m.freeze_all_network()

# torch.save(model, 'mnist_model.pt')

print('done')
