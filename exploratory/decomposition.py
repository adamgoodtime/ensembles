import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


gpu = True
# with torch.no_grad():
# torch.set_grad_enabled(False)
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

class torch_net(nn.Module):
    def __init__(self, input_size, layers, output_size):
        super(torch_net, self).__init__()

        l = np.hstack([input_size, layers, output_size])
        layers = [[l[i], l[i+1]] for i in range(len(l)-1)]

        self.conn = nn.ModuleList()
        for pre, post in layers:
            self.conn.append(nn.Linear(pre, post))

        self.relu = nn.ReLU()
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for layer in self.conn[:-1]:
            x = self.relu(layer(x))
        return self.logsoft(self.conn[-1](x))



input_size = 20
num_samples = 100
num_classes = 5
data = torch.Tensor(np.random.random((num_samples, input_size)).astype(np.float32))
labels = torch.Tensor(np.random.randint(0, num_classes, num_samples)).type(torch.long)
# labels = nn.functional.one_hot(torch.arange(0, num_samples) % num_classes)

layers = [10]
n_models = 3
models = nn.ModuleList()
optimizers = []
lr = 0.003
momentum = 0.9
epochs = 1
batch_size = 5

for i in range(n_models):
    models.append(torch_net(input_size, layers, num_classes))
    optimizers.append(optim.SGD(models[-1].parameters(), lr=lr, momentum=momentum))
loss_function = nn.NLLLoss()

# train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
for epoch in range(epochs):
    total_loss = np.zeros(n_models)
    for b in range(0, num_samples, batch_size):
        idx = torch.Tensor(range(num_samples)[b:b+batch_size]).to(dtype=torch.int16)
        # batch_in = torch.index_select(data, 0, idx)
        # batch_out = torch.index_select(labels, 0, idx)
        batch_in = data[b:b+batch_size]
        batch_out = labels[b:b+batch_size]
        for idx, (m, o) in enumerate(zip(models, optimizers)):
            output = m.forward(batch_in)
            loss = loss_function(output, batch_out)
            o.zero_grad()
            loss.backward()
            o.step()
            total_loss[idx] += loss.item()
    print("epoch:{}, loss:{}".format(epoch, total_loss / num_samples))

"""
bias =  E[(f - y)^2]
variance = E[(f - E[f))^2]
covariance = 1/(m(m-1)E[(f_i - E[f_i])(f_j - E[f_j])]

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 784)
        out = model(images)
        _, predicted = torch.max(out, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Testing accuracy: {} %'.format(100 * correct / total))
    
torch.save(model, 'mnist_model.pt')
"""
print("Done")


