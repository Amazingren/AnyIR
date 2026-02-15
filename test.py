import os
import argparse
import subprocess
from pathlib import Path

# Data processing
import numpy as np
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl

# Third-party evaluation metrics
import lpips

# Local imports
from net.anyir import AnyIR
from utils.dataset_utils import AnyDnTestDataset, AnyIRTestDataset, CDD11
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor

# Set environment variables
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

# Remove SLURM env variables to avoid issues with Lightning
if "SLURM_NTASKS" in os.environ:
    del os.environ["SLURM_NTASKS"]
    del os.environ["SLURM_JOB_NAME"]


class AnyIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = AnyIR()
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]


class AnyIRTester:
    """Testing framework for AnyIR model."""
    
    def __init__(self, opt):
        """Initialize tester with options."""
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        
        # Initialize LPIPS only if requested
        self.loss_fn_lpips = None
        if hasattr(opt, 'use_lpips') and opt.use_lpips:
            print("Initializing LPIPS for perceptual metrics...")
            self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
            self.loss_fn_lpips.eval()
        
        # Create output directory
        Path(opt.output_path).mkdir(exist_ok=True, parents=True)
    
        # Store all metrics for averaging
        self.all_metrics = {
            'denoise': [],
            'derain': [],
            'dehaze': [],
            'deblur': [],
            'enhance': []
        }

    def _load_model(self):
        """Load the AnyIR model from checkpoint."""
        print(f"Loading checkpoint: {self.opt.ckpt_path}")
        model = AnyIRModel().load_from_checkpoint(self.opt.ckpt_path).to(self.device)
        model.eval()
        return model
    
    def test_denoise(self, dataset, sigma):
        """Test denoising with specified noise level."""
        output_path = os.path.join(self.opt.output_path, f'denoise/{sigma}/')
        Path(output_path).mkdir(exist_ok=True, parents=True)
        
        dataset.set_sigma(sigma)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

        metrics = self._evaluate_dataset(testloader, output_path)
        
        # Store metrics for later averaging
        self.all_metrics['denoise'].append(metrics)
        
        # Build result string based on available metrics
        result_str = f"Denoising (sigma={sigma}): PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
        if 'lpips' in metrics:
            result_str += f", LPIPS: {metrics['lpips']:.4f}"
        print(result_str)
        
        return metrics
    
    def test_restoration(self, dataset, task):
        """Test other restoration tasks (derain, dehaze, deblur, enhance)."""
        output_path = os.path.join(self.opt.output_path, f'{task}/')
        Path(output_path).mkdir(exist_ok=True, parents=True)
        
        dataset.set_dataset(task)
        testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

        metrics = self._evaluate_dataset(testloader, output_path)
        
        # Store metrics for later averaging
        self.all_metrics[task].append(metrics)

        # Build result string based on available metrics
        result_str = f"{task.capitalize()}: PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
        if 'lpips' in metrics:
            result_str += f", LPIPS: {metrics['lpips']:.4f}"
        print(result_str)
        
        return metrics
    
    def _evaluate_dataset(self, dataloader, output_path=None):
        """Evaluate model on dataset and optionally save results."""
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        lpips_meter = AverageMeter() if self.loss_fn_lpips else None
    
        with torch.no_grad():
            for (names, degrad_patch, clean_patch) in tqdm(dataloader):
                # Extract image name from list
                name = names[0] if isinstance(names[0], str) else names[0][0]
                
                # Move data to device
                degrad_patch = degrad_patch.to(self.device)
                clean_patch = clean_patch.to(self.device)

                # Run model inference
                restored = self.model(degrad_patch)
                
                # Calculate metrics
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)
                
                # Calculate LPIPS only if enabled
                if self.loss_fn_lpips:
                    restored_lpips = restored.clamp(0, 1) * 2 - 1
                    clean_lpips = clean_patch.clamp(0, 1) * 2 - 1
                    lpips_value = self.loss_fn_lpips(restored_lpips, clean_lpips).mean().item()
                    lpips_meter.update(lpips_value, N)
                
                # Save output image if path provided
                # if output_path is not None:
                    # save_image_tensor(restored, os.path.join(output_path, f"{name}.png"))
        
    # Build metrics dictionary conditionally
        metrics = {
            'psnr': psnr_meter.avg,
            'ssim': ssim_meter.avg,
        }
        
        if self.loss_fn_lpips:
            metrics['lpips'] = lpips_meter.avg

        return metrics
    
    def _calculate_average_metrics(self):
        """Calculate average metrics across all degradation types."""
        all_psnr = []
        all_ssim = []
        all_lpips = []
        
        # Collect all metrics
        for degradation_type, metrics_list in self.all_metrics.items():
            if metrics_list:  # Only process degradation types that were tested
                for metrics in metrics_list:
                    all_psnr.append(metrics['psnr'])
                    all_ssim.append(metrics['ssim'])
                    if 'lpips' in metrics:
                        all_lpips.append(metrics['lpips'])
        
        # Calculate averages
        avg_metrics = {}
        if all_psnr:
            avg_metrics['psnr'] = sum(all_psnr) / len(all_psnr)
        if all_ssim:
            avg_metrics['ssim'] = sum(all_ssim) / len(all_ssim)
        if all_lpips and len(all_lpips) > 0:
            avg_metrics['lpips'] = sum(all_lpips) / len(all_lpips)
            
        return avg_metrics
    
    def _print_average_metrics(self):
        """Print the average metrics across all degradation types."""
        avg_metrics = self._calculate_average_metrics()
        
        if not avg_metrics:
            return
            
        print("\n" + "="*50)
        print("AVERAGE RESULTS ACROSS ALL DEGRADATIONS:")
        print("-"*50)
        
        result_str = f"Average PSNR: {avg_metrics.get('psnr', 0):.2f}, "
        result_str += f"Average SSIM: {avg_metrics.get('ssim', 0):.4f}"
        
        if 'lpips' in avg_metrics:
            result_str += f", Average LPIPS: {avg_metrics.get('lpips', 0):.4f}"
            
        print(result_str)
        print("="*50)
        
        return avg_metrics

    def run_all_tests(self):
        """Run tests based on the specified mode."""
        # Create datasets
        denoise_datasets = self._create_denoise_datasets()
        
        if self.opt.mode == 0 or self.opt.mode >= 5:
            print('<------ Testing Denoising: ------>')
            self._run_denoise_tests(denoise_datasets)
            
        if self.opt.mode == 1 or self.opt.mode >= 5:
            print('<------ Testing Deraining: ------>')
            self._run_derain_tests()
            
        if self.opt.mode == 2 or self.opt.mode >= 5:
            print('<------ Testing Dehazing: ------>')
            self._run_dehaze_tests()
            
        if self.opt.mode == 3 or self.opt.mode >= 6:
            print('<------ Testing Deblurring: ------>')
            self._run_deblur_tests()
            
        if self.opt.mode == 4 or self.opt.mode >= 6:
            print('<------ Testing Enhance: ------>')
            self._run_enhance_tests()
    
        # Print average metrics at the end
        self._print_average_metrics()

    def _create_denoise_datasets(self):
        """Create datasets for denoising tests."""
        denoise_datasets = []
        for split in self.opt.denoise_splits:
            split_path = os.path.join(self.opt.denoise_path, split)
            dataset = AnyDnTestDataset(argparse.Namespace(denoise_path=split_path))
            denoise_datasets.append((dataset, split))
        return denoise_datasets
    
    def _run_denoise_tests(self, denoise_datasets):
        """Run denoising tests for all datasets and noise levels."""
        for dataset, name in denoise_datasets:
            for sigma in [15, 25, 50]:
                print(f'Testing {name} with Sigma={sigma}...')
                self.test_denoise(dataset, sigma)
    
    def _run_derain_tests(self):
        """Run deraining tests."""
        for split in self.opt.derain_splits:
            print(f'Testing {split} rain streak removal...')
            split_path = os.path.join(self.opt.derain_path, split)
            derain_set = AnyIRTestDataset(
                argparse.Namespace(derain_path=split_path),
                task='derain',
                addnoise=False,
                sigma=15
            )
            self.test_restoration(derain_set, task="derain")
    
    def _run_dehaze_tests(self):
        """Run dehazing tests."""
        print('Testing SOTS outdoor dehazing...')
        # Reuse dataset from derain with different task
        split_path = os.path.join(self.opt.dehaze_path, self.opt.dehaze_splits[0])
        dehaze_set = AnyIRTestDataset(
            argparse.Namespace(dehaze_path=split_path),
            task='dehaze',
            addnoise=False,
            sigma=15
        )
        self.test_restoration(dehaze_set, task="dehaze")
    
    def _run_deblur_tests(self):
        """Run deblurring tests."""
        print('Testing Deblurring (GOPRO)...')
        for split in self.opt.deblur_splits:
            split_path = os.path.join(self.opt.gopro_path, split)
            deblur_set = AnyIRTestDataset(
                argparse.Namespace(gopro_path=split_path),
                task='deblur',
                addnoise=False,
                sigma=15,
            )
            self.test_restoration(deblur_set, task="deblur")
    
    def _run_enhance_tests(self):
        """Run low-light enhancement tests."""
        print('Testing Low-light Enhancement (LOL)...')
        for split in self.opt.enhance_splits:
            split_path = os.path.join(self.opt.enhance_path, split)
            enhance_set = AnyIRTestDataset(
                argparse.Namespace(enhance_path=split_path),
                task='enhance',
                addnoise=False,
                sigma=15,
            )
            self.test_restoration(enhance_set, task="enhance")


class CDD11Tester:
    """Testing framework for CDD11 dataset using AnyIR model."""
    
    def __init__(self, opt):
        """Initialize tester with options and subset."""
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
        
        # Extract subset from trainset name
        _, self.subset = opt.trainset.split("_", maxsplit=1)
        
        # Load model
        self.model = self._load_model()
        
        # Initialize LPIPS only if requested
        self.loss_fn_lpips = None
        if hasattr(opt, 'use_lpips') and opt.use_lpips:
            print("Initializing LPIPS for perceptual metrics...")
            self.loss_fn_lpips = lpips.LPIPS(net='alex').to(self.device)
            self.loss_fn_lpips.eval()
        
        # Create output directory
        output_path = os.path.join(self.opt.output_path, f'cdd11/{self.subset}/')
        Path(output_path).mkdir(exist_ok=True, parents=True)
        self.output_path = output_path
    
    def _load_model(self):
        """Load the AnyIR model from checkpoint."""
        print(f"Loading checkpoint: {self.opt.ckpt_path}")
        model = AnyIRModel().load_from_checkpoint(self.opt.ckpt_path).to(self.device)
        model.eval()
        return model
    
    def run_test(self):
        """Run tests on CDD11 dataset with the specified subset."""
        print(f'<------ Testing CDD11 ({self.subset}): ------>')
        
        # Create dataset
        dataset = CDD11(self.opt, split="test", subset=self.subset)
        
        # Create dataloader
        testloader = DataLoader(
            dataset, 
            batch_size=1, 
            pin_memory=True, 
            shuffle=False, 
            num_workers=0
        )
        
        # Evaluate on dataset
        metrics = self._evaluate_dataset(testloader, self.output_path)
        
        # Print results
        result_str = f"CDD11 ({self.subset}): PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
        if 'lpips' in metrics:
            result_str += f", LPIPS: {metrics['lpips']:.4f}"
        print(result_str)
        
        return metrics
    
    def _evaluate_dataset(self, dataloader, output_path=None):
        """Evaluate model on dataset and optionally save results."""
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        lpips_meter = AverageMeter() if self.loss_fn_lpips else None
    
        with torch.no_grad():
            for (names, degrad_patch, clean_patch) in tqdm(dataloader):
                # Extract image name from list
                name = names[0] if isinstance(names[0], str) else names[0][0]
                
                # Move data to device
                degrad_patch = degrad_patch.to(self.device)
                clean_patch = clean_patch.to(self.device)

                # Run model inference
                restored = self.model(degrad_patch)
                
                # Calculate metrics
                temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)
                
                # Calculate LPIPS only if enabled
                if self.loss_fn_lpips:
                    restored_lpips = restored.clamp(0, 1) * 2 - 1
                    clean_lpips = clean_patch.clamp(0, 1) * 2 - 1
                    lpips_value = self.loss_fn_lpips(restored_lpips, clean_lpips).mean().item()
                    lpips_meter.update(lpips_value, N)
                
                # Save output image if path provided
                if output_path:
                    save_image_tensor(restored, os.path.join(output_path, f"{name}.png"))
        
        # Build metrics dictionary conditionally
        metrics = {
            'psnr': psnr_meter.avg,
            'ssim': ssim_meter.avg,
        }
        
        if self.loss_fn_lpips:
            metrics['lpips'] = lpips_meter.avg

        return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AnyIR testing script")
    
    # Testing mode
    parser.add_argument('--mode', type=int, default=6,
                       help='Test mode: 0=denoise, 1=derain, 2=dehaze, 3=deblur, '
                            '4=enhance, 5=three tasks, 6=all five tasks')
    
    parser.add_argument('--trainset', default="AnyIR", 
                        help=["AnyIR", "CDD11_all", "CDD11_single", "CDD11_double", "CDD11_triple"])


    # Device settings
    parser.add_argument('--cuda', type=int, default=0,
                       help='CUDA device index')
    
    # Dataset paths
    parser.add_argument('--denoise_path', type=str, 
                       default="/dataset/low-level/test/denoise/",
                       help='Path to denoising test images')
    parser.add_argument('--derain_path', type=str,
                       default="/dataset/low-level/test/derain/",
                       help='Path to deraining test images')
    parser.add_argument('--dehaze_path', type=str,
                       default="/dataset/low-level/test/dehaze/",
                       help='Path to dehazing test images')
    parser.add_argument('--gopro_path', type=str,
                       default="/datasets/low-level/test/deblur/",
                       help='Path to deblurring test images')
    parser.add_argument('--enhance_path', type=str,
                       default="/datasets/low-level/test/enhance/",
                       help='Path to low-light enhancement test images')
    parser.add_argument('--cdd11_path', type=str, 
                       default="/datasets/low-level/cdd11/",
                       help='Path to CDD11 test images')

    # Output settings
    parser.add_argument('--output_path', type=str,
                       default="output/",
                       help='Output directory for restored images')
    parser.add_argument('--ckpt_name', type=str,
                       default="model.ckpt",
                       help='Checkpoint filename')
    
    parser.add_argument('--use_lpips', action='store_true',
                       help='Enable LPIPS metric calculation (slower but more perceptually accurate)')

    args = parser.parse_args()
    
    # Set additional parameters
    args.ckpt_path = os.path.join("train_ckpt", args.ckpt_name)
    args.denoise_splits = ["bsd68/"]
    args.derain_splits = ["Rain100L/"]
    args.deblur_splits = ["gopro/"]
    args.enhance_splits = ["lol/"]
    args.dehaze_splits = ["SOTS/"]
    
    return args

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
    
    if "CDD11" in args.trainset:
        # Create tester and run tests for CDD11
        tester = CDD11Tester(args)
        tester.run_test()

    elif "AnyIR" in args.trainset:
        # Create tester and run tests
        tester = AnyIRTester(args)
        tester.run_all_tests()
    else:
        raise ValueError(f"Unknown dataset: {args.trainset}")

if __name__ == '__main__':
    main()