# JAX Optimization for PAPR Reduction
This project implements the optimization algorithm for PAPR reduction in OFDM systems based on monotonic alignment.

# Features
- checkpoint resume
- MATLAB .mat input
- automatic result saving and restoring

# Project Structure
- `main.py`: main script for running experiments
- `gen_pam_qam.py`: QAM/PAM signal generation
- `gen_matrices.py`: matrix generation functions
- `diff_mono_align.py`: differentiable monotonic alignment module
- `data/`: input data or `.mat` files
- `results/`: output results
- `checkpoints/`: saved checkpoints
