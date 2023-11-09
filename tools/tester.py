import time
import torch
import os
from torch import nn
import torch.nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
import torchvision
import matplotlib.pyplot as plt
import shutil
import logging
from torch.utils.tensorboard import SummaryWriter
import tqdm
import datetime
import glob
import lpips
from .model import *
from lion_pytorch import Lion
from .losses import *
from .util import calculate_psnr, tensor2img, calculate_fpsnr
from torch import distributed as dist
from .dataset import *


class Tester:
    def __init__(self, rank, world_size, args, distributed=True):
        print(rank, world_size, args)
        self.args = args
        self.rank = rank
        self.init_writer()
        self.log(f"rank {self.rank}")
        self.log(f"loading configs {self.args}", level="LOG")

        torch.manual_seed(42)
        import random

        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.benchmark = True
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.log(f"model {self.model}")

        self.test()
        self.destroy()

    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.log("Initializing distributed")
        # os.environ['MASTER_ADDR'] = self.args.distributed_addr
        # os.environ['MASTER_PORT'] = self.args.distributed_port
        # dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if world_size == 1:
            dist.init_process_group(
                "gloo",
                init_method=f"file:///tmp/gloo{time.time()}",
                rank=rank,
                world_size=world_size,
            )
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def find_free_port(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            return str(s.getsockname()[1])

    def init_datasets(self):
        psis = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        self.test_dataloaders = dict()
        for subset in self.args.datasets_to_use_test:
            dataset = subset
            if subset != "FFHQH":
                dataset = "IhdDataset"
            test_dataset = globals()[dataset](
                is_train=False, subset=subset, opt=self.args.datasets
            )
            self.log(
                f"Initializing dataset {len(test_dataset)}", level="LOG")
            test_dataloader = DataLoader(
                dataset=test_dataset,
                sampler=DistributedSampler(test_dataset, rank=self.rank),
                batch_size=1,
                num_workers=self.args.num_workers,
                persistent_workers=True,
            )
            self.test_dataloaders[subset] = test_dataloader

    def init_model(self, model_decoder="UnetDecoder"):
        self.log("Initializing model")

        self.model = PHNet(
            enc_sizes=self.args.model.enc_sizes,
            skips=self.args.model.skips,
            grid_count=self.args.model.grid_counts,
            init_weights=self.args.model.init_weights,
            init_value=self.args.model.init_value,
        ).to(self.rank)

        self.log(
            f"Restoring from checkpoint: {self.args.checkpoint_dir}", level="LOG")
        self.load_checkpoint(self.args.checkpoint_dir)

        self.model_ddp = DDP(
            self.model,
            device_ids=[self.rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            weights = glob.glob("checkpoints/*/*")
            weights.sort(key=os.path.getmtime)
            checkpoint = weights[-1]

        self.model.load_state_dict(
            torch.load(checkpoint, map_location=f"cuda:{self.rank}"), strict=False
        )

    def init_writer(self):
        if self.rank == 0:
            self.timestamp = f"{datetime.datetime.now():%d_%B_%H_%M}"
            # self.logger = logging.getLogger()
            # self.logger.setLevel(logging.INFO)
            # fh = logging.FileHandler(f"{self.args.checkpoint_dir}/{self.timestamp}/logs.log")
            # self.logger.addHandler(fh)
            print("LOGGER CREATED")
            self.log("Initializing writer")
            self.writer = SummaryWriter(
                f"{self.args.log_dir}/test_{self.args.experiment_name}_{self.timestamp}"
            )
            # logging.basicConfig(filename=f"{self.args.checkpoint_dir}/{self.timestamp}/log.txt", level=logging.INFO)

    def normalize(self, x):
        return 2 * (x - x.min() / x.max() - x.min()) - 1

    def log_image_grid(self, step, **kwargs):
        for name, tensor in kwargs.items():
            image_grid = torchvision.utils.make_grid(
                tensor[: self.args.log_image_number, ::].detach().cpu()
            )
            self.writer.add_image(f"{name}", image_grid, step)

    def test(self):
        for subset in self.test_dataloaders:
            test_dataloader = self.test_dataloaders[subset]
            self.log(f'testing on {subset}', level="LOG")
            self.log(f"Number of steps: {len(test_dataloader)}", level="LOG")

            if self.rank == 0:
                self.model_ddp.eval()
                total_loss, total_count = 0, 0
                psnr_scores = 0
                fpsnr_scores = 0
                mse_scores = 0
                mse_scores_img = 0
                fmse_scores = 0

                fmse_scores_ratio = {"5": 0, "15": 0, "100": 0}
                mse_scores_ratio = {"5": 0, "15": 0, "100": 0}
                ratio_count = {"5": 0, "15": 0, "100": 0}

                with torch.no_grad():
                    with autocast(enabled=not self.args.disable_mixed_precision):
                        for i, input_dict in enumerate(tqdm.tqdm(test_dataloader)):
                            inputs = input_dict["inputs"].to(self.rank)
                            composite = input_dict["comp"].to(self.rank)
                            real = input_dict["real"].to(self.rank)
                            mask = input_dict["mask"].to(self.rank)
                            harmonized = self.model_ddp(composite, mask)
                            blending = mask * harmonized + (1 - mask) * composite
                            blending_img = tensor2img(blending, bit=8)
                            real_img = tensor2img(real, bit=8)
                            psnr_score = calculate_psnr(blending_img, real_img)
                            fore_area = torch.sum(mask)
                            mse_score = torch.nn.functional.mse_loss(
                                blending, real)
                            mse_score_img = torch.nn.functional.mse_loss(
                                torch.from_numpy(blending_img).float(),
                                torch.from_numpy(real_img).float(),
                            )
                            fmse_score = (
                                tensor2img(blending * mask -
                                           real * mask, bit=9) ** 2
                            ).sum() / fore_area
                            fpsnr_score = calculate_fpsnr(fmse_score)

                            psnr_scores += psnr_score
                            fpsnr_scores += fpsnr_score
                            mse_scores += mse_score
                            mse_scores_img += mse_score_img
                            fmse_scores += fmse_score

                            batch_size = inputs.shape[0]
                            total_count += batch_size
                            if i % self.args.log_image_interval == 0:
                                self.log_image_grid(
                                    i,
                                    alpha=mask,
                                    inputs=inputs,
                                    real=real,
                                    composite=composite,
                                    predicted=harmonized,
                                    predicted_blend=blending,
                                    real_masked=real * mask,
                                    predicted_mask=harmonized * mask,
                                )
                psnr_scores_mu = psnr_scores / total_count
                fpsnr_scores_mu = fpsnr_scores / total_count
                mse_score_mu = mse_scores / total_count
                fmse_score_mu = fmse_scores / total_count
                mse_score_img_mu = mse_scores_img / total_count

                for k in ratio_count.keys():
                    mse_scores_ratio[k] /= ratio_count[k] + 1e-8
                    fmse_scores_ratio[k] /= ratio_count[k] + 1e-8
                # avg_loss = total_loss / total_count
                self.log(f"Dataset: {subset}, Test set results:", level="LOG")
                self.log(
                    f"Test set psnr score: {psnr_scores_mu}", level="LOG")
                self.writer.add_scalar(f"{subset} psnr", psnr_scores_mu, i)
                self.log(
                    f"Test set fpsnr score: {fpsnr_scores_mu}", level="LOG")
                self.writer.add_scalar(f"{subset} fpsnr", fpsnr_scores_mu, i)
                self.log(
                    f"Test set MSE score: {mse_score_img_mu}", level="LOG")
                self.writer.add_scalar(f"{subset} mse", mse_score_img_mu, i)
                self.log(
                    f"Test set fMSE score: {fmse_score_mu}", level="LOG")
                self.writer.add_scalar(f"{subset} fmse", fmse_score_mu, i)
                self.log(
                    f"PSNR: {psnr_scores_mu}, FPSNR: {fpsnr_scores_mu}, mse img: {mse_score_img_mu}",
                    level="LOG",
                )

    def destroy(self):
        dist.destroy_process_group()

    def log(self, msg, level="INFO"):
        if self.rank == 0:
            print(f"[GPU{self.rank}] {msg}")
        # if self.rank == 0 and level == "LOG":
        #     self.logger.info(f'[GPU{self.rank}] {msg}')
