# run_combined.py
import torch
from train_utils import train_with_momentum, test
from init_utils import base_parse_args, task_init, logger_init
from fastDP import PrivacyEngine
import argparse
import warnings

# Import the combined optimizer
from combined_optimizer import DPLinearMomentumOptimizer

if __name__ == '__main__':
    """
    Main entry point for running Differentially Private training with Per-sample Momentum and Linear Filtering.
    
    This script:
    1. Parses command line arguments
    2. Initializes training components (model, data loaders, etc.)
    3. Reads filter coefficients from a specified file
    4. Sets up the optimizer with linear momentum and DP components
    5. Runs the training loop with differential privacy
    """
    warnings.filterwarnings("ignore")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Combined DP-SGD with linear momentum')
    parser = base_parse_args(parser)
    
    # Add momentum specific arguments
    parser.add_argument('--momentum_length', type=int, default=2, help='number of history steps for inner momentum')
    parser.add_argument('--coef_file', default='./coefs/a9b1.csv', type=str, help='coefficients')
    parser.add_argument('--inner_momentum', default=0.1, type=float, help='per-sample momentum coefficients')
    args = parser.parse_args()

    # Initialize training
    train_dl, test_dl, model, device, sample_size, acc_step, noise = task_init(args)
    log_file = logger_init(args, noise, sample_size//args.mnbs)

    # Read coefficients from file
    with open(args.coef_file, "r") as f:
        coefs = f.readlines()
        a = [float(i) for i in coefs[0].split(",") if i.strip()]
        b = [float(i) for i in coefs[1].split(",") if i.strip()]

    # Initialize optimizer
    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer = DPLinearMomentumOptimizer(
        model.parameters(),
        optimizer=base_optimizer,
        inner_k0=args.momentum_length,
        inner_gamma = args.inner_momentum,
        a=a,
        b=b
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Initialize DP engine
    privacy_engine = PrivacyEngine(
        model, 
        noise_multiplier=noise,
        max_grad_norm=args.clipping_norm,
        sample_size=sample_size,
        batch_size=args.bs,
        epochs=args.epoch
    )
    privacy_engine.attach(optimizer)

    # Training loop  
    for epoch in range(args.epoch):
        train_with_momentum(
            model, train_dl, optimizer, criterion, log_file, 
            device=device, epoch=epoch, log_frequency=args.log_freq,
            acc_step=acc_step
        )
    test(model, test_dl, criterion, log_file, device=device, epoch=epoch)