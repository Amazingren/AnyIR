import argparse

parser = argparse.ArgumentParser()

# --- Training Parameters ---
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=130, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=32,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
parser.add_argument("--num_gpus",type=int,default=1,help = "Number of GPUs to use for training")
parser.add_argument('--trainset', default="AnyIR", help=["AnyIR", "CDD11_all", "CDD11_single", "CDD11_double", "CDD11_triple"])

# --- Degradation Types ---
parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')

# --- Loss Weights ---
parser.add_argument('--fft_loss_weight', type=float, default=0.1, help='FFT loss weight.')


# --- Training Dataset Paths ---
parser.add_argument('--data_file_dir', type=str, default='./data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/dataset/low-level/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/dataset/low-level/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/dataset/low-level/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--gopro_dir', type=str, default='/datasets/low-level/Train/Deblur/',
                    help='where clean images of denoising saves.')
parser.add_argument('--enhance_dir', type=str, default='/datasets/low-level/Train/Enhance/',
                    help='where clean images of denoising saves.')
parser.add_argument('--cdd11_path', type=str, default='/datasets/low-level/Train/CDD/',
                    help='where clean images of denoising saves.')

# --- Output and Checkpoint Paths ---
parser.add_argument('--output_path', type=str, default=None, help='output imgs save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="anyir",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="train_ckpt/",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--resume_from",type=str,default=None,help = "Path to the checkpoint to resume training from")


options = parser.parse_args()
