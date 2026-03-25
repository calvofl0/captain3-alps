# Running CAPTAIN at CSCS on Alps

[CAPTAIN](https://github.com/captain-project/captain3preview) is a reinforcement learning system for optimizing conservation and restoration strategies in space and time. This repository contains the setup for running it on the CSCS Alps cluster. This setup leverages containerization with Podman/Enroot and the SLURM workload manager for training and for performing inference.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
  - [1. Container Build](#1-container-build)
  - [2. Environment Configuration](#2-environment-configuration)
  - [3. Job Configuration](#3-job-configuration)
- [Running the Job](#running-the-job)
- [Understanding the Configuration](#understanding-the-configuration)
- [Troubleshooting](#troubleshooting)

## Overview

This setup uses:
- **Base Image**: NVIDIA PyTorch 24.06 container (with CAPTAIN additions)
- **Container Engine**: CSCS Container Engine (Enroot) with EDF (Environment Description Files)
- **Scheduler**: SLURM
- **Training Framework**: PyTorch

## Prerequisites

- Access to CSCS Alps cluster
- Working directory with sufficient storage (preferably on `/capstor` or `/iopsstor`)
- Basic familiarity with SLURM and containerization

## Quick Start

For those who just want to run:

```bash
# 1. Build the container (one-time setup)
podman build -t ngc-pytorch-captain3preview:24.06 .
mkdir -p ${SCRATCH}/captain3
enroot import -x mount -o ${SCRATCH}/captain3/ngc-pytorch-captain3preview-24.06.sqsh podman://ngc-pytorch-captain3preview:24.06

# 2. Edit the environment file
#   - This repository is assumed to be cloned in ${HOME}/captain3-alps
#   - Update '${HOME}/captain3-alps' paths in ngc-pytorch-captain3preview-24.06.toml otherwise

# 3. Configure your job
# Prepare a SLURM script for running your job. You can edit the examples found in the examples folder; at the very least you need to:
#   - Replace '<YOUR_ACCOUNT>' with your project id at CSCS
#   - Replace '<YOUR_EMAIL>' with your email address

# 4. Submit the job
#   - Plot the example data
sbatch examples/captain3plot.sbatch
#   - Run inference from a pre-trained model
sbatch examples/captain3inference.sbatch
#   - Train a model from the example data
sbatch examples/captain3train.sbatch
#   - Run inference after model training with the example data
sbatch examples/captain3inference_after_training.sbatch
```

## Detailed Setup

### 1. Container Build

The base NVIDIA PyTorch container needs to be customized for running CAPTAIN and converted to a format compatible with the CSCS Container Engine.

#### Build the Podman container:

```bash
podman build -t ngc-pytorch-captain3preview:24.06 .
```

This reads the `Containerfile` in the working directory and builds a container tagged as `ngc-pytorch-captain3preview:24.06`.

#### Convert to SquashFS format:

```bash
enroot import -x mount -o ${SCRATCH}/captain3/ngc-pytorch-captain3preview-24.06.sqsh podman://ngc-pytorch-captain3preview:24.06
```

This command:
- Converts the Podman container to SquashFS format (`.sqsh`)
- SquashFS is a compressed read-only filesystem ideal for container images
- The resulting file can be mounted directly without extraction, saving disk space and improving startup time

**Note**: The `-x mount` flag keeps the SquashFS file mounted during import, and `-o` specifies the output filename.

### 2. Environment Configuration

The Environment Description File (EDF) `ngc-pytorch-captain3preview-24.06.toml` tells the CSCS Container Engine how to run your container. You might need to update the working directory paths therein.

#### Edit `ngc-pytorch-captain3preview-24.06.toml`:

Find these lines and replace `${HOME}/captain3-alps` with your actual working directory path containing this repository:

```toml
mounts = [
    "/capstor",
    "/iopsstor",
    "/dev/shm/${USER}",
    "${HOME}/.ssh",
    "${HOME}/captain3-alps"  # <-- Update this
] 
workdir = "${HOME}/captain3-alps"  # <-- Update this
```

**Example**: If this repository has been cloned to `/capstor/projects/csstaff/enoether/captain3-alps`, change both occurrences to that path and change your working directory (`cd`) to that same path

#### Understanding the EDF Configuration:

The EDF file configures several important aspects:

- **image**: Points to the SquashFS container image
- **mounts**: Directories from the host system that will be accessible inside the container
  - `/capstor` and `/iopsstor`: High-performance storage filesystems
  - `/dev/shm/${USER}`: Shared memory for the user
  - `${HOME}/.ssh`: SSH keys for potential remote operations
  - Your working directory: Where your code and data reside
  
- **annotations**: Enable AWS OFI NCCL plugin for optimized GPU communication
  - `com.hooks.aws_ofi_nccl.enabled = "true"`: Activates the plugin
  - `com.hooks.aws_ofi_nccl.variant = "cuda12"`: Specifies CUDA 12 compatibility

- **Environment variables**:
  - `NCCL_DEBUG = "INFO"`: Enables detailed NCCL logging for debugging
  - `CUDA_CACHE_DISABLE = "1"`: Disables CUDA kernel cache to avoid issues
  - `TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"`: Enables async error handling for better error reporting
  - `MPICH_GPU_SUPPORT_ENABLED = "0"`: Disables MPI GPU support (using NCCL instead)

### 3. Job Configuration

The SLURM batch scripts inside the `examples` folder are recipies for the several kind of tasks you might want to perform.

#### Required edits:

```bash
#SBATCH --account <YOUR_ACCOUNT>  # Replace with your project id at CSCS
#SBATCH --mail-user <YOUR_EMAIL>  # Replace with your email address
```

#### Understanding the SLURM directives:

The script includes several SLURM directives that control job execution:

```bash
#SBATCH --job-name=captain3-inference     # Job name in queue
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # GPUs per node (Alps has 4 GPUs/node)
#SBATCH --cpus-per-task=1                 # CPU cores per GPU task
#SBATCH --time=00:30:00                   # Maximum runtime (30 minutes)
#SBATCH --partition=debug                 # Pick the most appropriate partition
#SBATCH --account=<YOUR_ACCOUNT>          # Your allocation account
#SBATCH --mail-type=END,FAIL              # Email on job end or failure
```

**Important**: The total number of GPUs used will be `nodes × ntasks-per-node`.

Extensive SLURM documentation specific to the Alps clusters can be found here: https://docs.cscs.ch/running/slurm/.

## Running the Job

Once everything is configured:

```bash
sbatch examples/captain3<sample_task>.sbatch
```

Replace `<sample_task>` with the task you want to perform, in order to match one of the available SLURM scripts.

### Monitoring your job:

```bash
# Check job status
squeue -u $USER

# View job output (while running or after completion)
tail -f logs/captain3<sample_task>-<jobid>.out

# Cancel a job if needed
scancel <jobid>
```

## Understanding the Configuration

### Container Engine on Alps

The CSCS Container Engine uses a customized version of NVIDIA Pyxis to integrate containers with SLURM. Key features:

- **Automatic image caching**: Remote images are cached in `${SCRATCH}/.edf_imagestore`
- **Shared containers**: All tasks on a node share the same container instance
- **Registry authentication**: Configure in `~/.config/enroot/.credentials` for private registries
- **EDF search path**: EDFs can be placed in `~/.edf/` and referenced by name (without `.toml` extension)

To use the EDF by name instead of path:

```bash
mkdir -p ~/.edf
cp ngc-pytorch-captain3preview-24.06.toml ~/.edf/
# Then in your sbatch script, use: --environment=ngc-pytorch-captain3preview-24.06
```

## Troubleshooting

### Container build fails

```bash
# Check if you have podman installed and configured
podman --version

# Ensure you have sufficient disk space
quota
df -h .
```

### Enroot import fails

```bash
# Verify the container exists in podman
podman images

# Check for sufficient disk space
quota
df -h .
```

### Job fails immediately

```bash
# Check the SLURM output file for errors
cat logs/captain3<sample_task>-<jobid>.out

# Common issues:
# - Wrong account name
# - Insufficient allocation hours
# - Invalid partition name
```

### Out of Memory (OOM) errors

If you encounter OOM errors:

**Reduce model size**: Consider using a smaller model for initial testing

### NCCL/Communication errors

If you see NCCL timeout or communication errors:

1. **Check interconnect**: Ensure AWS OFI NCCL plugin is enabled in the EDF
2. **Increase timeout**: Add `NCCL_TIMEOUT=1800` to the environment section of the EDF
3. **Verify GPU connectivity**: Run `nvidia-smi topo -m` on a compute node to check GPU topology
4. **Check for hardware issues**: Some nodes may have faulty GPUs or network adapters

### Container runtime errors

```bash
# Verify the EDF file syntax
cat ngc-pytorch-captain3preview-24.06.toml

# Check if mounts are accessible
ls -la /capstor /iopsstor

# Ensure the container image path is correct
ls -la ${SCRATCH}/captain3/ngc-pytorch-captain3preview-24.06.sqsh
```

### Permission errors

```bash
# Verify you have write access to output directories

# Check ownership of working directory
ls -la $(dirname $PWD)
```

## Additional Resources

- [CAPTAIN v.3.0 - Preview](https://github.com/captain-project/captain3preview)
- [CSCS Container Engine Documentation](https://docs.cscs.ch/software/container-engine/run/)
- [NVIDIA PyTorch Container Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
- [Alps System Documentation](https://docs.cscs.ch/)

## Support

For issues specific to:
- **CSCS infrastructure**: Contact CSCS support
- **This setup**: Review the troubleshooting section above
