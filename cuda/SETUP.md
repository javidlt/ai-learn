# CUDA Learning Environment Setup

This guide will help you set up a cloud-based CUDA development environment using Vast.ai.

## Prerequisites

### Windows Users
- **PuTTY**: Install [PuTTY](https://www.putty.org/) for SSH connections
- **PuTTY SCP (PSCP)**: For file transfers (included with PuTTY installer)

### All Platforms
- Basic understanding of terminal/command line
- A Vast.ai account (create one at [vast.ai](https://vast.ai))

## Setup Instructions

### 1. Create a GPU Instance

1. Navigate to the [NVIDIA CUDA template](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=NVIDIA%20CUDA)
2. **Recommended Configuration**:
   - GPU: 1x RTX 3060 Ti (good price/performance ratio)
   - RAM: At least 8GB
   - Storage: 20GB minimum
   - CUDA Version: 11.8 or higher

3. Click "Rent" on your selected machine

### 2. Connect to Your Instance

Once the instance is running:

1. Click the **"Open"** button next to your instance
2. From the dropdown menu, select **"Open Jupyter Terminal"**
3. Alternatively, you can SSH directly:
   ```bash
   ssh -p [PORT] root@[IP_ADDRESS]
   ```

### 3. Verify CUDA Installation

Run these commands to verify your setup:

```bash
# Check CUDA version
nvcc --version

# Check GPU information
nvidia-smi

# Test a simple CUDA compilation
cd ~/cuda/basics
nvcc hello_world.cu -o hello_world
./hello_world
```

## Project Structure

```
cuda/
├── SETUP.md           # This file
└── basics/
    └── hello_world.cu # Your first CUDA program
```

## Useful Commands

- **Compile CUDA code**: `nvcc filename.cu -o output`
- **Run compiled program**: `./output`
- **Monitor GPU usage**: `watch -n 1 nvidia-smi`
- **Check CUDA samples**: `/usr/local/cuda/samples/`

## Troubleshooting

### Connection Issues
- Verify your instance is running (green status on Vast.ai)
- Check firewall settings if direct SSH fails
- Use the web-based Jupyter terminal as fallback

### Compilation Errors
- Ensure CUDA toolkit is in PATH: `echo $PATH | grep cuda`
- Check nvcc is accessible: `which nvcc`
- Verify GPU compute capability matches code requirements

## Next Steps

1. Explore basic CUDA kernels in `basics/hello_world.cu`
2. Learn about thread organization (blocks, grids, threads)
3. Practice memory management (host ↔ device transfers)
4. Experiment with parallel algorithms

## Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Vast.ai Documentation](https://vast.ai/docs/)