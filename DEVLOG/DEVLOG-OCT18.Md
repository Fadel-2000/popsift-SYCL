# Development Log - October 18, 2025
## Enabling NVIDIA GPU Support for SYCL in PopSift

**Author:** Mohammad Fadel  
**Date:** October 18, 2025  
**System:** Ubuntu, Intel Core i5-12450HX, NVIDIA GeForce RTX 4050 Laptop GPU  
**Compiler:** Intel oneAPI DPC++ 2024.2.0 → Custom Intel LLVM with CUDA backend

---

## 1. Problem Statement

### Initial Issue
Intel oneAPI DPC++ compiler (`icpx`) could **not detect NVIDIA GPU** on the system.

**Symptoms:**
- Running `sycl-ls` only showed Intel CPU OpenCL runtime:
  ```
  [opencl:cpu][opencl:0] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i5-12450HX
  ```
- No CUDA backend available
- No NVIDIA GPU device visible to SYCL applications
- PopSift application defaulted to CPU execution

**Hardware Verification:**
```bash
$ lspci | grep -i vga
00:02.0 VGA compatible controller: Intel Corporation Alder Lake-S [UHD Graphics]
01:00.0 VGA compatible controller: NVIDIA Corporation AD107M [GeForce RTX 4050 Max-Q / Mobile]

$ nvidia-smi
NVIDIA GeForce RTX 4050 Laptop GPU
Driver Version: 580.65.06
CUDA Version: 13.0
```

**Conclusion:** GPU hardware working, but SYCL runtime cannot access it due to missing CUDA backend in packaged Intel oneAPI.

---

## 2. Root Cause Analysis

### Why Intel oneAPI Couldn't See NVIDIA GPU

The standard Intel oneAPI Base Toolkit (2024.2.0) does **not** include CUDA backend support by default. The compiler only supports:
- Intel GPUs (via Level Zero)
- OpenCL devices (CPU and Intel GPUs)

To enable NVIDIA GPU support, one of the following is required:
1. **Codeplay oneAPI for NVIDIA GPUs plugin** (commercial/registration required)
2. **Building Intel LLVM from source with CUDA backend enabled**

We attempted the Codeplay plugin download but encountered:
- Broken download links (404 errors)
- Empty installer files (0 bytes)

Therefore, we proceeded with **Option 2: Building from source**.

---

## 3. Solution: Build Intel LLVM with CUDA Backend

### 3.1. Install Build Dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 git
```

**Packages installed:**
- `build-essential` - GCC/G++ compilers
- `cmake` - Build system generator
- `ninja-build` - Fast build tool
- `python3` - Required for Intel's build scripts
- `git` - Version control

---

### 3.2. Clone Intel LLVM Repository

```bash
cd ~
git clone https://github.com/intel/llvm -b sycl
cd llvm
```

**Repository:** https://github.com/intel/llvm  
**Branch:** `sycl` (Intel's SYCL implementation)  
**Size:** ~2-3 GB

---

### 3.3. Configure with CUDA Backend

```bash
python3 buildbot/configure.py \
    --cuda \
    --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda" \
    --cmake-opt="-DCMAKE_BUILD_TYPE=Release"
```

**Configuration flags:**
- `--cuda` - **Enable NVIDIA GPU support via CUDA backend**
- `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda` - Path to CUDA installation
- `-DCMAKE_BUILD_TYPE=Release` - Optimized build (not debug)

**Note:** CUDA was installed at `/usr/bin/nvcc` (not `/usr/local/cuda`), so the actual root was `/usr`.

**CMake configuration verified:**
```
SYCL_ENABLE_BACKENDS=level_zero_v2;cuda;level_zero;opencl
CUDA found: /usr (version 12.0.140)
```

---

### 3.4. Build Intel LLVM (30-60 minutes)

```bash
python3 buildbot/compile.py -j$(nproc)
```

**Build time:** ~45 minutes on 12-core Intel i5-12450HX  
**Output location:** `~/llvm/build/bin/clang++`  
**Key files created:**
- `~/llvm/build/bin/clang++` - SYCL compiler with CUDA support
- `~/llvm/build/lib/libur_adapter_cuda.so` - CUDA adapter library
- `~/llvm/build/lib/libsycl.so` - SYCL runtime library

**Build verification:**
```bash
$ ~/llvm/build/bin/clang++ --version
clang version 22.0.0
Target: x86_64-unknown-linux-gnu
```

---

### 3.5. Fix Runtime Library Path

**Problem:** After building, `sycl-ls` still didn't show the GPU because the CUDA adapter library wasn't in the library search path.

```bash
$ ~/llvm/build/bin/sycl-ls
[opencl:cpu][opencl:0] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i5-12450HX
# CUDA backend not loaded!
```

**Solution:** Add the custom LLVM library path to `LD_LIBRARY_PATH`:

```bash
# Temporary (current session)
export LD_LIBRARY_PATH=~/llvm/build/lib:$LD_LIBRARY_PATH

# Permanent (add to ~/.bashrc)
echo 'export LD_LIBRARY_PATH=~/llvm/build/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Verification - SUCCESS! ✓**
```bash
$ ~/llvm/build/bin/sycl-ls
[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA GeForce RTX 4050 Laptop GPU 8.9 [CUDA 13.0]
[opencl:cpu][opencl:0] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i5-12450HX OpenCL 3.0
```

**NVIDIA GPU is now visible to SYCL!** 

---

## 4. Configure PopSift for Custom LLVM Compiler

### 4.1. Update Root CMakeLists.txt

**File:** `CMakeLists.txt` (project root)

**Problem:** Original CMakeLists.txt only accepted Intel's `icpx` compiler and rejected custom Clang builds.

**Solution:** Modified compiler check to accept both Intel oneAPI and custom LLVM:

```cmake
# ============================================================================
# Compiler Check - Accept Intel icpx OR custom LLVM with SYCL support
# ============================================================================
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR 
        CMAKE_CXX_COMPILER MATCHES "icpx" OR 
        CMAKE_CXX_COMPILER MATCHES "dpcpp" OR
        (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER MATCHES "llvm")))
    message(FATAL_ERROR 
        "\n=================================================================\n"
        "ERROR: This project requires SYCL compiler!\n"
        "Current compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})\n\n"
        "Please use one of:\n"
        "  Intel oneAPI:  cmake .. -DCMAKE_CXX_COMPILER=icpx\n"
        "  Custom LLVM:   cmake .. -DCMAKE_CXX_COMPILER=$HOME/llvm/build/bin/clang++\n"
        "=================================================================\n")
endif()

message(STATUS "✓ Using SYCL compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
```

---

### 4.2. Add NVIDIA GPU Target Flags

Added conditional compilation flags for NVIDIA GPU when using custom LLVM:

```cmake
# Add NVIDIA GPU target (CUDA backend) - only if using custom LLVM
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER MATCHES "llvm")
    message(STATUS "Configuring SYCL for NVIDIA RTX 4050 Laptop GPU (sm_89)...")
    
    # Add NVIDIA GPU target
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
    
    # Target RTX 4050 Laptop (Ada Lovelace, sm_89)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89")
    
    # Linker flags
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
endif()
```

**GPU Architecture Selection:**
- RTX 4050 Laptop = Ada Lovelace architecture = **sm_89** (compute capability 8.9)

---

### 4.3. Build PopSift with CUDA Support

```bash
cd ~/Desktop/popsift-SYCL/build
rm -rf *

cmake .. \
    -DCMAKE_CXX_COMPILER=$HOME/llvm/build/bin/clang++ \
    -DCMAKE_C_COMPILER=$HOME/llvm/build/bin/clang \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

**CMake output:**
```
✓ Using SYCL compiler: /home/mohammad-fadel/llvm/build/bin/clang++
  Compiler ID: Clang
Configuring SYCL for NVIDIA RTX 4050 Laptop GPU (sm_89)...
-- CXX Flags: -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89 -std=c++17
-- Linker Flags: -fsycl -fsycl-targets=nvptx64-nvidia-cuda
```

**Build result:**  SUCCESS
```
[100%] Built target popsift
[100%] Built target popsift-demo
[100%] Built target popsift-match
```

---

## 5. Verification and Testing

### 5.1. Verify Device Selection

```bash
cd ~/Desktop/popsift-SYCL/build/Linux-x86_64
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
export CUDA_LAUNCH_BLOCKING=1  # Synchronous kernel launches for debugging
```

**Environment variables:**
- `ONEAPI_DEVICE_SELECTOR=cuda:gpu` - Force SYCL to use CUDA GPUs
- `CUDA_LAUNCH_BLOCKING=1` - Synchronous execution for better error messages

---

### 5.2. Run PopSift Demo

```bash
./popsift-demo -i apple.pgm
```

**Output:**
```
PopSift version: 1.0.0
apple.pgm
uploading byte image (30x20 pixels)

Octave 0 queue initialized on: NVIDIA GeForce RTX 4050 Laptop GPU
Octave 0 allocating on device: NVIDIA GeForce RTX 4050 Laptop GPU
Octave 0 allocated: data=0x74275b000000, dog=0x74275b01c400

Octave 1 queue initialized on: NVIDIA GeForce RTX 4050 Laptop GPU
Octave 1 allocating on device: NVIDIA GeForce RTX 4050 Laptop GPU
Octave 1 allocated: data=0x74275b028000, dog=0x74275b02f400

Octave 2 queue initialized on: NVIDIA GeForce RTX 4050 Laptop GPU
Octave 2 allocating on device: NVIDIA GeForce RTX 4050 Laptop GPU
Octave 2 allocated: data=0x74275b032400, dog=0x74275b034400
```

**✅ GPU Execution Confirmed!**
- SYCL queues created on NVIDIA GPU
- Device memory allocated successfully
- Kernels launching on GPU

---

## 6. Current Status and Known Issues

### 6.1. What's Working ✓

| Component | Status |
|-----------|--------|
| Intel LLVM with CUDA backend |  Built successfully (clang 22.0.0) |
| CUDA adapter library |  `libur_adapter_cuda.so` loaded |
| GPU device detection |  NVIDIA GPU visible in `sycl-ls` |
| PopSift compilation |  Builds with CUDA target (sm_89) |
| SYCL queue creation |  Queues created on NVIDIA GPU |
| Device memory allocation |  `sycl::malloc_device()` working |
| Basic kernel launch |  Kernels submit to GPU |

---

### 6.2. Known Issues

**Kernel Execution Crash:**
```
CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
Function: enqueueKernelLaunch
```

**Location:** First pyramid building kernel (`horiz_from_input_image` in `s_pyramid_build_ra.cc`)

**Cause:** Memory bounds checking issue in kernel implementation. The kernel accesses memory outside allocated bounds during linear interpolation.

**Not a compiler/backend issue** - this is a standard GPU programming bug that needs kernel-level debugging and fixing.

---

## 7. Key Learnings

### 7.1. Intel oneAPI SYCL Limitations

- **Standard Intel oneAPI does NOT support NVIDIA GPUs out-of-the-box**
- Requires either:
  - Codeplay oneAPI for NVIDIA GPUs plugin (commercial, registration required)
  - Building Intel LLVM from source with `--cuda` flag

### 7.2. SYCL Compilation Targets

```bash
# Compile for NVIDIA GPU (CUDA backend)
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
        -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89 \
        source.cpp -o executable
```

### 7.3. Runtime Configuration

```bash
# Set library path for custom LLVM
export LD_LIBRARY_PATH=~/llvm/build/lib:$LD_LIBRARY_PATH

# Force CUDA device selection
export ONEAPI_DEVICE_SELECTOR=cuda:gpu

# Enable synchronous kernel execution (debugging)
export CUDA_LAUNCH_BLOCKING=1

# Enable SYCL tracing
export SYCL_UR_TRACE=1
```

---

## 8. Next Steps

### 8.1. Fix Kernel Memory Access Issues
- Debug `horiz_from_input_image` kernel in `s_pyramid_build_ra.cc`
- Add proper bounds checking for array indices
- Fix linear interpolation logic to prevent out-of-bounds access

### 8.2. Complete Kernel Migration
- Verify all other kernels work correctly on NVIDIA GPU
- Add error checking and validation for each kernel
- Performance profiling and optimization

### 8.3. Documentation
- Update README with NVIDIA GPU build instructions
- Document runtime environment variables
- Add troubleshooting guide for common CUDA backend issues

---

## 9. References

- **Intel LLVM SYCL:** https://github.com/intel/llvm
- **SYCL Specification:** https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
- **CUDA Compute Capabilities:** https://developer.nvidia.com/cuda-gpus
- **Codeplay oneAPI for NVIDIA:** https://developer.codeplay.com/products/oneapi/nvidia/

---

## 10. Files Modified

| File | Changes |
|------|---------|
| `CMakeLists.txt` (root) | Added custom LLVM compiler support, NVIDIA GPU target flags |
| `src/popsift/sift_octave.h` | Added SYCL queue member, accessor methods |
| `src/popsift/sift_octave.cc` | Implemented `initQueue()`, updated `alloc()` and `free()` |
| `src/popsift/sift_pyramid.h` | Changed `dogs_from_blurred()` return type to `sycl::event` |
| `src/popsift/s_pyramid_build.cc` | Converted DoG kernel to SYCL `parallel_for` |
| `src/popsift/common/plane_base.h` | Added SYCL-aware memory allocation |
| `src/popsift/common/plane_base.cc` | Implemented `sycl::malloc_device()` allocation |

---

**End of Development Log - October 18, 2025**