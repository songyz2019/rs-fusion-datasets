# understand the code of joint classification of HSI and DSM data in about 50 lines
# if you're using v0.12.3 and before, you should remove ClassificationMapper and its related code to make it work
import skimage
import torch
from torch.nn import Sequential, LazyConv2d, ReLU, LazyBatchNorm2d, Module, LazyLinear
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from rs_fusion_datasets import Houston2013, Trento, Muufl, ClassificationMapper

class MyModel(Module):
    def __init__(self, n_class):
        super().__init__()
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
    optimizer = Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        for hsi,dsm,lbl,info in DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True):
            optimizer.zero_grad()
            y_hat = model(hsi, dsm)
            loss = cross_entropy(y_hat, lbl)
            loss.backward()
            optimizer.step()
            print(f"{epoch=} {loss=}")
    torch.save(model.state_dict(), 'model.pth')

    # Test
    testset = Houston2013('test', patch_size=5)
    benchmarker = testset.benchmarker()
    model.eval()
    for hsi,dsm,lbl,info in DataLoader(testset, batch_size=1):
        y_hat = model(hsi, dsm)
        benchmarker.add_sample(info['location'], y_hat, lbl)
    ca,oa,aa,kappa = benchmarker.ca(), benchmarker.oa(), benchmarker.aa(), benchmarker.kappa()
    print(f"CA: {ca.round(decimals=5)}")
    print(f"OA: {oa:.5f}, AA: {aa:.5f}, Kappa: {kappa:.5f}")
    skimage.io.imsave('result_test.png', benchmarker.predicted_image(format='hwc'))

    # Draw The full predicted image
    fullset = Houston2013('full', patch_size=5)
    benchmarker = fullset.benchmarker()
    model.eval()
    i_batch = 0
    for hsi,dsm,_,info in DataLoader(fullset, batch_size=64, shuffle=False):
        y_hat = model(hsi, dsm)
        benchmarker.add_sample(info['location'], y_hat)
        i_batch += 1
        print(f"drawing {i_batch*64}/{len(fullset)} pixels", end='\r')
    skimage.io.imsave('result_full.png', benchmarker.predicted_image(format='hwc'))
    print("the result images are saved as result_test.png and result_full.png")
