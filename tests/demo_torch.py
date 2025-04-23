# understand the code of joint classification of HSI and DSM data in 50 lines
import torch
from torch import argmax
from torch.nn import Sequential, LazyConv2d, ReLU, LazyBatchNorm2d, Module, LazyLinear
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from rs_fusion_datasets import Houston2013, Trento, Muufl

class MyModel(Module):
    def __init__(self, n_class):
        self.conv_hsi = Sequential(
            LazyConv2d(64, kernel_size=3), ReLU(), LazyBatchNorm2d(),
            LazyConv2d(16, kernel_size=3)
        )
        self.conv_dsm = Sequential(
            LazyConv2d(8,  kernel_size=3), ReLU(), LazyBatchNorm2d(),
            LazyConv2d(16, kernel_size=3)
        )
        self.classifier = LazyLinear(n_class)
    
    def forward(self, hsi, dsm):
        x = self.conv_hsi(hsi) + self.conv_dsm(dsm)
        return self.classifier(x.flatten(start_dim=1))

if __name__=='__main__':
    # Train
    trainset = Houston2013('train', patch_size=5)
    model = MyModel(n_class=trainset.INFO['n_class'])
    optimizer = Adam(model.parameters(), lr=0.005)
    for epoch in range(10):
        for hsi,dsm,lbl,info in DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True):
            y_hat = model(hsi, dsm)
            loss = cross_entropy(y_hat, lbl)
            loss.backward()
            optimizer.step()
            print(f"{epoch=} {loss=}")
    torch.save(model.state_dict(), 'model.pth')

    # Test
    testset = Houston2013(subset='test', patch_size=5)
    n_correct = 0
    model.eval()
    for hsi,dsm,lbl,info in DataLoader(testset, batch_size=1):
        y_hat = model(hsi, dsm)
        if argmax(y_hat, dim=1) == argmax(lbl, dim=1):
            n_correct += 1
            print(f"accuracy: {n_correct}/{len(testset)}")
    
