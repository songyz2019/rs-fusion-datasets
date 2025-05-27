import random
from pathlib import Path
from datetime import datetime

import numpy
import numpy as np
import torch
from torch.nn import Sequential, LazyConv2d, ReLU, LazyBatchNorm2d, Module, LazyLinear
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as T

from rs_fusion_datasets import AugsburgOuc

class MyModel(Module):
    def __init__(self, n_class):
        super().__init__()
        self.conv_hsi = Sequential(
            LazyConv2d(64, kernel_size=3), ReLU(), LazyBatchNorm2d(),
            LazyConv2d(32, kernel_size=3), ReLU(), LazyBatchNorm2d(),
            LazyConv2d(16, kernel_size=3)
        )
        self.conv_dsm = Sequential(
            LazyConv2d(8,  kernel_size=3), ReLU(), LazyBatchNorm2d(),
            LazyConv2d(12,  kernel_size=3), ReLU(), LazyBatchNorm2d(),
            LazyConv2d(16, kernel_size=3)
        )
        self.classifier = LazyLinear(n_class)
    
    def forward(self, hsi, dsm, lbl):
        x = self.conv_hsi(hsi) + self.conv_dsm(dsm)
        y_hat = self.classifier(x.flatten(start_dim=1))
        loss = cross_entropy(y_hat, lbl)
        return y_hat, loss

class AutoTrainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.trainset   = AugsburgOuc('train', patch_size=9)
        self.valset     = AugsburgOuc('test',  patch_size=9)
        self.testset    = AugsburgOuc('full',  patch_size=9)
        self.model      = MyModel(n_class=self.trainset.n_class).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9)

        self.model_name = self.model.__class__.__name__ + '_' + self.trainset.uid
        self.acc_best   = -1
        self.summary    = SummaryWriter(log_dir=f"runs/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")      # Tensorboard记录器

        self.transform_hsi = T.Compose([
            T.ToDtype(torch.float),
        ])
        self.transform_dsm = T.Compose([
            T.ToDtype(torch.float),
        ])
        self.transform_lbl = T.Compose([
            T.ToDtype(torch.float)
        ])
        
    def train(self, i_epoch):
        self.model.train()
        for i, (hsi, dsm, lbl, _) in enumerate(DataLoader(self.trainset, shuffle=True, batch_size=64, drop_last=False, num_workers=4)):
            hsi = self.transform_hsi(hsi).to(self.device)
            dsm = self.transform_dsm(dsm).to(self.device)
            lbl = self.transform_lbl(lbl).to(self.device)
            self.optimizer.zero_grad()

            y_hat, loss = self.model(hsi, dsm, lbl)
            loss.backward()
            self.optimizer.step()

            print(f"epoch={i_epoch} batch={i} {loss=:.5f}", end='\r')
            self.summary.add_scalar("train/loss", loss, i_epoch*100+i)
            if i_epoch==0 and i==0:
                self.summary.add_images("train/hsi", self.trainset.hsi2rgb(hsi))
                self.summary.add_images("train/dsm", dsm[:,0:1, :, :])
        self.scheduler.step()
        self.summary.add_scalar("train/lr", self.scheduler.get_last_lr()[0], i_epoch)
        print("")

    @torch.no_grad()
    def val(self, i_epoch):
        self.model.eval()
        benchmarker = self.valset.benchmarker()
        for i, (hsi, dsm, lbl, ext) in enumerate(DataLoader(self.valset, shuffle=False, batch_size=64, num_workers=4, drop_last=False)):
            # Forward
            hsi = self.transform_hsi(hsi).to(self.device)
            dsm = self.transform_dsm(dsm).to(self.device)
            lbl = self.transform_lbl(lbl).to(self.device)
            y_hat, loss = self.model(hsi, dsm, lbl)
            benchmarker.add_sample(ext['location'], y_hat, lbl)
            print(f"epoch={i_epoch} batch={i}", end='\r')
            self.summary.add_scalar("val/loss", loss, i_epoch*100+i)
            if i==0 and i_epoch==0:
                self.summary.add_images("val/hsi", self.valset.hsi2rgb(hsi))
                self.summary.add_images("val/dsm", dsm)


        # Logging
        _txt = lambda name,content: self.summary.add_text(  'val/'+name, str(content), global_step=i_epoch)
        _fig = lambda name,content: self.summary.add_figure('val/'+name, content, global_step=i_epoch)
        _chw = lambda name,content: self.summary.add_image( 'val/'+name, content, global_step=i_epoch, dataformats='CHW')
        _num = lambda name,content: self.summary.add_scalar('val/'+name, content, global_step=i_epoch)
        ca, oa, aa, kappa = benchmarker.ca(), benchmarker.oa(), benchmarker.aa(), benchmarker.kappa()
        _num('oa',   oa      )
        _num('aa',   aa      )
        _num('kappa',kappa   )
        _txt('frac', benchmarker.frac()    )
        _txt('conf', benchmarker.confusion_matrix)
        if oa >= self.acc_best:
            self.acc_best = oa
            _txt("oa",    oa )
            _txt("aio4paper", benchmarker.aio4paper())
            # _fig("conf",      benchmarker.confusion_plot()) # The fig doesn't work in windows
            _chw("lbl_prd",   benchmarker.predicted_image())
            _chw("lbl_err",   benchmarker.error_image(underlying=self.valset.hsi2rgb()))
            torch.save(self.model.state_dict(), Path(f'runs/{self.model_name}.pt'))
        print(f"epoch={i_epoch} {oa=:.3f} {aa=:.3f} {kappa=:.3f} best_oa={self.acc_best:.3f}")

    @torch.no_grad()
    def test(self):
        self.model.load_state_dict(torch.load(Path(f'runs/{self.model_name}.pt'), weights_only=False))
        self.model.eval()
        benchmarker = self.testset.benchmarker()
        for i, (hsi, dsm, lbl, ext) in enumerate(DataLoader(self.testset, shuffle=False, batch_size=256, num_workers=4)):
            # Forward
            hsi = self.transform_hsi(hsi).to(self.device)
            dsm = self.transform_dsm(dsm).to(self.device)
            lbl = self.transform_lbl(lbl).to(self.device)

            y_hat, _ = self.model(hsi, dsm, lbl)
            # 记录
            print(f"batch={i}", end='\r')
            benchmarker(ext['location'], y_hat)
            if i % 200 == 0: self.summary.add_image("test/full_y_hat", benchmarker.predicted_image())
        self.summary.add_image("test/full_y_hat", benchmarker.predicted_image())
        print("")
        self.benchmarker = benchmarker

    def start(self, n_epoch, pretrained :Path | None = None):
        # Load pretrained model if specified
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained, weights_only=True))
            self.summary.add_text("train/pretrained", str(pretrained))
            self.val(0)

        # Log basic information of datasets to Tensorboard
        for name, dataset in zip(['train', 'val', 'test'],[self.trainset, self.valset, self.testset]):
            self.summary.add_image(f"dataset/{name}_hsi", dataset.hsi2rgb())
            self.summary.add_image(f"dataset/{name}_lbl", dataset.lbl2rgb())
            for i in range(dataset.dsm.shape[0]):
                self.summary.add_image(f"dataset/{name}_dsm_{i}", dataset.dsm[i:i+1])
            self.summary.add_text(f"dataset/{name}_size", str(len(dataset)))
            self.summary.add_text(f"dataset/{name}_uid", dataset.uid)

        # Train, validate and test
        for i_epoch in range(n_epoch):
            self.train(i_epoch)
            if (i_epoch+1) % 5 == 0:
                self.val(i_epoch)
        self.test()
        self.summary.flush()


def main():
    print('started')
    # Set random seed
    seed = 0x00
    torch.set_printoptions(threshold=np.inf)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float)

    # Start
    trainer = AutoTrainer()
    trainer.start(n_epoch=100)


if __name__ == '__main__':
    main()