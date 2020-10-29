#You need to install the following python packages
#pytorch, vit_pytorch.
import torch
import torchvision
from vit_pytorch import ViT
import time
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(97)

Dpath = '/data/mnist'
Bs_Train = 100
Bs_Test = 1000

tform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,),(0.3081,))])

tr_set = torchvision.datasets.MNIST(Dpath, train = True, download = True,
                                       transform = tform_mnist)

tr_load = torch.utils.data.DataLoader(tr_set, batch_size = Bs_Train, shuffle = True)

ts_set = torchvision.datasets.MNIST(Dpath, train = False, download = True, transform = tform_mnist)

ts_load = torch.utils.data.DataLoader(ts_set, batch_size = Bs_Test, shuffle = True)

def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    
    for i, (data, target) in enumerate(data_load):
        optimz.zero_grad()
        out = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()
        
        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())

def evaluate(model, data_load, loss_val):
    model.eval()
    
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for data, target in data_load:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            tloss += loss.item()
            csamp += pred.eq(target).sum()

    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    
N_EPOCHS = 25

start_time = time.time()
model = ViT(image_size=28, patch_size=4, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
optimz = optim.Adam(model.parameters(), lr=0.003)

trloss_val, tsloss_val = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_iter(model, optimz, tr_load, trloss_val)
    evaluate(model, ts_load, tsloss_val)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')