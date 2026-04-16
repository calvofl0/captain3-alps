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
- [CAPTAIN development](#captain-development)
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
export XDG_RUNTIME_DIR=/run/user/$UID
export XDG_DATA_HOME=$XDG_RUNTIME_DIR
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

#### Make sure Podman has read-write access to a runtime directory in a supported filesystem

```bash
export XDG_RUNTIME_DIR=/run/user/$UID
export XDG_DATA_HOME=$XDG_RUNTIME_DIR
```

#### Build the Podman container:

```bash
podman build -t ngc-pytorch-captain3preview:24.06 .
```

This reads the `Containerfile` in the working directory and builds a container tagged as `ngc-pytorch-captain3preview:24.06`.

#### Convert to SquashFS format:

```bash
mkdir -p ${SCRATCH}/captain3
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

## CAPTAIN development

In the previous sections we saw how to run jobs in production. The Python virtual environment containing the CAPTAIN code and even the data were included into the built container, facilitating reproducibility and pipeline-sharing between people and also computing infrastructure.

The build process was however long, and in a development context rebuilding the container each time the source code is modified is unrealistic. In this section we will create a minimal container with just the system dependencies required to run Python and CAPTAIN on GPUs.

The virtual environment hosting CAPTAIN and all Python dependencies will be located outside of the container, but it will be created and run with all the context of the container.

CAPTAIN itself will be installed in the virtual environment in *editable mode*, which will allow to modify the CAPTAIN source code without the need of re-installing CAPTAIN each time in the virtual environment.

### 1. Clone and patch the CAPTAIN code

This needs to be made only once, so that the development source code of CAPTAIN is available on the cluster. This source code can then be modified without rebuilding the container.

```bash
git clone https://github.com/captain-project/captain3preview ~/captain3preview
patch -f -p1 -d ~/captain3preview < ./captain3preview.patch
sed -i -e 's/numpy[><=.,0-9]*/numpy>=1.23/' -e 's/\(requires-python[[:blank:]]*=[[:blank:]]*"\).*\("\)/\1>=3.10\2/' \
 -e '/^[[:blank:]]*"torch[<>=,.0-9]*",*[[:blank:]]*$/d' \
 -e '/\[\[tool\.uv\.index\]\]/,/^$/d' \
 -e '/\[tool\.uv\.sources\]/,/^$/d' \
 -e '$a\\n[tool.uv]\noverride-dependencies = [\n    "torch; sys_platform == '"'"'never'"'"'",\n]' ~/captain3preview/pyproject.toml
```

> [!WARNING]
> The location of the source code in `~/captain3preview` can be changed, but then the corresponding bind path in `ngc-pytorch-24.06.toml` needs to be adjusted accordingly, so that the source code is also available from within then container.

> [!NOTE]
> Patching the code might no more be required when using a newer version of the `ngc-pytorch` container; the container version must however be compatible with the NVIDIA driver version reported by `nvidia-smi`. Check the release notes for each container version [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html). Version 24.06 is, at the time of writing, the most recent version guaranteed to be compatible with all clusters at CSCS.

### 2. Container Build

We need to build the base NVIDIA PyTorch container with a few extra packages for running CAPTAIN and convert it to a format compatible with the CSCS Container Engine.

#### Make sure Podman has read-write access to a runtime directory in a supported filesystem

```bash
export XDG_RUNTIME_DIR=/run/user/$UID
export XDG_DATA_HOME=$XDG_RUNTIME_DIR
```

#### Build the Podman container:

```bash
podman build -t ngc-pytorch:24.06 -f Containerfile.pytorch-24.06
```

This reads `Containerfile.pytorch-24.06` in the working directory and builds a container tagged as `ngc-pytorch:24.06`.

#### Convert to SquashFS format:

```bash
enroot import -x mount -o ${SCRATCH}/captain3/ngc-pytorch-24.06.sqsh podman://ngc-pytorch:24.06
```

### 3. Prepare the Python virtual environment to run CAPTAIN

#### Start a session within the container

```bash
srun -A <YOUR_PROJECT> --environment ./ngc-pytorch-24.06.toml --pty bash
```

#### Create the Python virtual environment

```bash
uv venv --python "$(which python3)" --system-site-packages --seed --relocatable --link-mode=copy .venv
```

#### Install build/install dependencies

```bash
uv pip install --python .venv/bin/python -c ./constraints.txt setuptools hatchling editables
```

#### Install CAPTAIN in editable mode
```bash
uv pip install --python .venv/bin/python --no-build-isolation -c ./constraints.txt -e ~/captain3preview
```

#### Quit the session and revoke the job allocation
```bash
exit
```

> [!IMPORTANT]
> Whenever additional packages need to be installed in the Python virtual environment, it needs to be done from within the container. This is achieved by starting a session as demonstrated above, with the `srun` command.

### 4. Run jobs in the development container

The SLURM script `examples/captain3plot-dev.sbatch` can be used as a template for submitting jobs that use the development container. It is simpler than the production version, since both CAPTAIN and all data is outside of the container.

However, all folders containing data used for the runs need to be bind mounted inside the container: check the `mounts` entry in the environment file `ngc-pytorch-24.06.toml`.

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
