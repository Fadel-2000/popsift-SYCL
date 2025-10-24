# PopSift CUDA to SYCL Migration Guide

## Table of Contents
1. [Overview](#overview)
2. [Build System Changes](#build-system-changes)
3. [Source Code Changes](#source-code-changes)
4. [CUDA to SYCL API Mapping](#cuda-to-sycl-api-mapping)
5. [Known Issues and Limitations](#known-issues-and-limitations)
6. [Testing and Verification](#testing-and-verification)

---

## Overview

This document details all modifications made to migrate PopSift from CUDA --> CPU(C++) to SYCL using Intel DPC++ compiler (oneAPI 2024.2).

**Migration Summary:**
- **Target Platform**:
- **Compiler**: Intel DPC++ (icpx)
- **SYCL Version**: SYCL 2020
- **C++ Standard**: C++17 (upgraded from C++11)
- **Key Changes**: Memory management, kernel launches, build system

---

## Build System Changes

### 1. Root CMakeLists.txt

**File**: `/CMakeLists.txt`

#### Changes Made:

##### 1.1 Compiler Validation
Added mandatory check to enforce Intel DPC++ compiler:

```cmake
# ============================================================================
# CRITICAL: Force Intel DPC++ Compiler Check
# ============================================================================
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND 
   NOT CMAKE_CXX_COMPILER MATCHES "icpx" AND 
   NOT CMAKE_CXX_COMPILER MATCHES "dpcpp")
    message(FATAL_ERROR 
        "\n=================================================================\n"
        "ERROR: This project requires Intel DPC++ compiler (icpx)!\n"
        "Current compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})\n\n"
        "Please reconfigure with:\n"
        "  cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx\n"
        "=================================================================\n")
endif()

message(STATUS "✓ Using Intel DPC++ compiler: ${CMAKE_CXX_COMPILER}")
```

##### 1.2 SYCL Compiler Flags
Added SYCL compilation and linking flags:

```cmake
# ============================================================================
# SYCL Compiler Setup (BEFORE C++ STANDARD)
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsycl")
```

##### 1.3 C++ Standard Upgrade
Upgraded from C++11 to C++17 (SYCL requirement):

```cmake
# ============================================================================
# C++ Standard - Required for SYCL (MUST BE AFTER -fsycl FLAG)
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add explicit -std=c++17 flag AFTER -fsycl
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
```

**Build Command:**
```bash
cd ~/Desktop/popsift-SYCL/build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

### 2. Application CMakeLists.txt

**File**: `src/application/CMakeLists.txt`

#### Changes Made:

##### 2.1 Removed CUDA Dependencies

**Before (CPU):**
```cmake
set(PD_LINK_LIBS       ${Boost_LIBRARIES} ${CUDA_CUDADEVRT_LIBRARY})
```

**After (SYCL):**
```cmake
set(PD_LINK_LIBS       ${Boost_LIBRARIES})
```

**Reason**: SYCL doesn't need CUDA device runtime library. Linking is handled by `-fsycl` flag.

##### 2.2 Disabled OpenImageIO

**Before (CPU):**
```cmake
find_package(OpenImageIO)

if(OpenImageIO_FOUND)
  message(STATUS "OpenImageIO found")
  set(PD_COMPILE_OPTIONS "-DUSE_OIIO")
else()
  message(WARNING "OpenImageIO not found -- Falling back to pgmread")
  set(PD_COMPILE_OPTIONS "" )
endif()
```

**After (SYCL):**
```cmake
# Disable OpenImageIO - incompatible with Intel SYCL compiler
# find_package(OpenImageIO)
message(WARNING "OpenImageIO disabled (incompatible with SYCL) -- Using pgmread")
set(PD_COMPILE_OPTIONS "" )
```

**Reason**: OpenImageIO bundles `fmt` library with `long double` incompatibilities in Intel DPC++ compiler.

**Error Without This Change:**
```
/usr/include/OpenImageIO/detail/fmt/format.h:3122:38: error: 
implicit instantiation of undefined template 'fmt::detail::dragonbox::float_info<long double>'
```

##### 2.3 Upgraded C++ Standard to C++17

**Before (CPU):**
```cmake
add_executable(popsift-demo  main.cpp pgmread.cpp pgmread.h)
set_property(TARGET popsift-demo PROPERTY CXX_STANDARD 11)

add_executable(popsift-match match.cpp pgmread.cpp pgmread.h)
set_property(TARGET popsift-match PROPERTY CXX_STANDARD 11)
```

**After (SYCL):**
```cmake
add_executable(popsift-demo  main.cpp pgmread.cpp pgmread.h)
set_property(TARGET popsift-demo PROPERTY CXX_STANDARD 17)
set_property(TARGET popsift-demo PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET popsift-demo PROPERTY CXX_EXTENSIONS OFF)

add_executable(popsift-match match.cpp pgmread.cpp pgmread.h)
set_property(TARGET popsift-match PROPERTY CXX_STANDARD 17)
set_property(TARGET popsift-match PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET popsift-match PROPERTY CXX_EXTENSIONS OFF)
```

**Reason**: SYCL 2020 requires minimum C++17.

**Error Without This Change:**
```
error: static assertion failed due to requirement '201402L >= 201703L': 
DPCPP does not support C++ version earlier than C++17.
```

##### 2.4 Removed OpenImageIO from Targets

**Before (CPU):**
```cmake
target_include_directories(popsift-demo PUBLIC PopSift::popsift
                                               OpenImageIO::OpenImageIO ${PD_INCLUDE_DIRS})
target_link_libraries(popsift-demo PUBLIC PopSift::popsift
                                          OpenImageIO::OpenImageIO ${PD_LINK_LIBS})
```

**After (SYCL):**
```cmake
target_include_directories(popsift-demo PUBLIC ${PD_INCLUDE_DIRS})
target_link_libraries(popsift-demo PUBLIC PopSift::popsift ${PD_LINK_LIBS})
```

---

## Source Code Changes

### 3. Application Source Files

#### 3.1 main.cpp

**File**: `src/application/main.cpp`

**Changes**: Wrapped OpenImageIO includes with conditional compilation guards.

**Before (CPU):**
```cpp

#include <OpenImageIO/imageio.h>

#include "pgmread.h"

using namespace OIIO;
using namespace std;
```

**After (SYCL):**
```cpp

#ifdef USE_OIIO
#include <OpenImageIO/imageio.h>
#endif

#include "pgmread.h"

#ifdef USE_OIIO
using namespace OIIO;
#endif
using namespace std;
```

**Impact**: Since `USE_OIIO` is not defined, OpenImageIO code is excluded from compilation. Only PGM file format is supported.

---

### 4. Core Library Changes

#### 4.1 plane_2d.h

**File**: `src/popsift/common/plane_2d.h`

**Changes**: Added SYCL header and new SYCL-aware allocation method.

##### 4.1.1 Added SYCL Header Include

**Change:**
```cpp
#include <sycl/sycl.hpp>  // NEW: Added for SYCL queue support
```

---

##### 4.1.2 Added SYCL-Aware Allocation Method

**New method added:**
```cpp
// NEW: SYCL-aware alloc (device memory)
inline void alloc( int w, int h, int d, sycl::queue* queue ) {
    _ptr.reset( new PlaneT<T> );
    _ptr->alloc( w, h, d, queue );
}
```

**Purpose**: Provides an overload that accepts a SYCL queue pointer for device memory allocation.

---

##### 4.1.3 Added Device Pointer Access Methods

**New methods added:**
```cpp
// NEW: Device pointer access for SYCL kernels
inline T* getDevicePtr() {
    return _ptr ? _ptr->getDevicePtr() : nullptr;
}

inline const T* getDevicePtr() const {
    return _ptr ? _ptr->getDevicePtr() : nullptr;
}

// NEW: Get pitch in elements (not bytes)
inline int getPitch() const {
    return _ptr ? _ptr->getPitchElements() : 0;
}
```


**Purpose**: 
- `getDevicePtr()`: Returns raw device pointer for SYCL kernel access
- `getPitch()`: Returns pitch in elements for proper 2D memory indexing

---


**Note**: The actual CUDA to SYCL memory management happens in **`plane_base.h` and `plane_base.cc`**, which implement the `PlaneT<T>` class that `PlaneD` wraps. The `plane_2d.h` file only provides the high-level wrapper interface.

#### 4.2 plane_base.h

**File**: `src/popsift/common/plane_base.h`

**Changes**: Added SYCL header, new allocation method, and device pointer access methods.

##### 4.2.1 Added SYCL Header Include

**Change:**
```cpp
#include <sycl/sycl.hpp>  // NEW: Added for SYCL queue support
```

---

##### 4.2.2 Added New Member Variables to PlaneBase

**Before (CPU):**
```cpp
private:
    void* _plane; /// the plane
    int   _pitch; /// width in bytes with padding
    int   _x;     /// width in elements
    int   _y;     /// height
    int   _z;     /// depth
};
```

**After (SYCL):**
```cpp
private:
    void* _plane; /// the plane
    int   _pitch; /// width in bytes with padding
    int   _x;     /// width in elements
    int   _y;     /// height
    int   _z;     /// depth
    MemMode      _mode{AlignmentUndefined};  // NEW: Track allocation mode
    sycl::queue* _queue{};                   // NEW: Store queue for device memory
};
```


**Purpose**: 
- `_mode`: Track whether memory is host or device allocated
- `_queue`: Store SYCL queue pointer for proper deallocation

---

##### 4.2.3 Added SYCL-Aware Allocation Method to PlaneBase

**New method declaration:**
```cpp
// NEW: SYCL-aware alloc (device memory with queue)
bool alloc( int elemSize, int w, int h, int d, sycl::queue* queue );
```

---

##### 4.2.4 Added Device Pointer Access Methods to PlaneBase

**New methods added:**
```cpp
// NEW: Get device pointer
inline void* getDevicePtr() {
    return _plane;
}

inline const void* getDevicePtr() const {
    return _plane;
}
```

**Purpose**: Provide typed access to device memory pointer for SYCL kernels.

---

##### 4.2.5 Added SYCL-Aware Methods to PlaneT Template

**New method in PlaneT template:**
```cpp
// NEW: SYCL-aware alloc (calls PlaneBase with queue)
inline void alloc( int w, int h, int d, sycl::queue* queue )
{
    PlaneBase::alloc( elemSize(), w, h, d, queue );
}
```
---

**New device pointer access methods in PlaneT:**
```cpp
// NEW: Typed device pointer access
inline T* getDevicePtr() {
    return static_cast<T*>(PlaneBase::getDevicePtr());
}

inline const T* getDevicePtr() const {
    return static_cast<const T*>(PlaneBase::getDevicePtr());
}

// NEW: Get pitch in elements (not bytes)
inline int getPitchElements() const {
    return PlaneBase::getPitch() / elemSize();
}
```

**Purpose**:
- `getDevicePtr()`: Returns typed device pointer for kernel access
- `getPitchElements()`: Returns pitch in elements instead of bytes for easier indexing

---

##### 4.2.6 Summary of Changes

**Actual changes to plane_base.h:**

1. **Line ~18**: Added `#include <sycl/sycl.hpp>`

2. **Lines ~65**: Added new `alloc()` overload in PlaneBase:
   ```cpp
   bool alloc( int elemSize, int w, int h, int d, sycl::queue* queue );
   ```

3. **Lines ~73-80**: Added device pointer accessors in PlaneBase:
   ```cpp
   inline void* getDevicePtr() { return _plane; }
   inline const void* getDevicePtr() const { return _plane; }
   ```

4. **Lines ~121**: Added member variables in PlaneBase:
   ```cpp
   MemMode      _mode{AlignmentUndefined};
   sycl::queue* _queue{};
   ```

5. **Lines ~155**: Added SYCL-aware `alloc()` in PlaneT:
   ```cpp
   inline void alloc( int w, int h, int d, sycl::queue* queue ) { ... }
   ```

6. **Lines ~167-177**: Added typed device pointer accessors in PlaneT:
   ```cpp
   inline T* getDevicePtr() { ... }
   inline const T* getDevicePtr() const { ... }
   inline int getPitchElements() const { ... }
   ```

---


**Note**: The heavy implementation work happens in `plane_base.cc`, which implements the new `alloc(int, int, int, int, sycl::queue*)` and updated `dealloc()` methods.

#### 4.3 plane_base.cc

**File**: `src/popsift/common/plane_base.cc`

**Changes**: Implemented SYCL-aware allocation and deallocation.

##### 4.3.1 SYCL-Aware Allocation (New Implementation)

```cpp
// NEW: SYCL-aware alloc implementation
bool PlaneBase::alloc( int elemSize, int w, int h, int d, sycl::queue* queue )
{
    if (!queue) {
        POP_FATAL("PlaneBase::alloc() called with null queue pointer");
        return false;
    }

    _pitch = w * elemSize;
    _x     = w;
    _y     = h;
    _z     = d;
    _queue = queue;  // Store queue for later deallocation

    size_t sz = static_cast<size_t>(w) * h * d * elemSize;

    try {
        // Allocate device memory using SYCL
        _plane = sycl::malloc_device(sz, *queue);
        
        if (!_plane) {
            stringstream ss;
            ss << "Failed to allocate " << sz << " bytes of device memory via SYCL.";
            POP_FATAL(ss.str());
            return false;
        }
        
        _mode = MemMode::CUDA1D;  // Mark as device memory
        return true;
    }
    catch (const sycl::exception& e) {
        stringstream ss;
        ss << "SYCL exception during device allocation: " << e.what();
        POP_FATAL(ss.str());
        return false;
    }
}
```

**Features**:
- Validates queue pointer (nullptr check)
- Uses `sycl::malloc_device()` for device memory
- Exception handling for SYCL errors
- Stores queue pointer for deallocation
- Marks memory mode as device memory

##### 4.3.2 Updated Deallocation

**Before (CPU):**
```cpp
void PlaneBase::dealloc()
{
    if (!_plane) {
        free(_plane);
        _plane = nullptr;
    }
}
```

**After (SYCL):**
```cpp
void PlaneBase::dealloc()
{
    if (!_plane) return;

    // Check if this is device memory allocated via SYCL
    if (_mode == MemMode::CUDA1D && _queue) {
        try {
            sycl::free(_plane, *_queue);
        }
        catch (const sycl::exception& e) {
            cerr << "SYCL exception during deallocation: " << e.what() << endl;
        }
    }
    else {
        // Regular host memory
        free(_plane);
    }
    
    _plane = nullptr;
    _queue = nullptr;
    _mode = MemMode::AlignmentUndefined;
}
```

**Features**:
- Distinguishes between device and host memory (`_mode` check)
- Uses `sycl::free()` for device memory
- Exception handling for SYCL errors
- Resets all pointers and mode

---

#### 4.4 sift_pyramid.h

**File**: `src/popsift/sift_pyramid.h`

**Changes**: Updated `dogs_from_blurred()` method signature to return SYCL event.

##### 4.4.1 Method Signature Change

**Before (CPU):**
```cpp
void dogs_from_blurred( int octave, int max_level );
```

**After (SYCL):**
```cpp
sycl::event dogs_from_blurred( int octave, int max_level );
```

**Purpose**: Return `sycl::event` to allow for asynchronous operation tracking and synchronization in SYCL.

---


#### 4.6 sift_octave.h

**File**: `src/popsift/sift_octave.h`

**Changes**: Added SYCL queue infrastructure to Octave class.

##### 4.6.1 Added SYCL Header Include

**Change:**
```cpp
#include <sycl/sycl.hpp>  // NEW: Added for SYCL queue support
```
---

##### 4.6.2 Added SYCL Queue Member Variable

**Before (CPU):**
```cpp
class Octave
{
    int   _w{};
    int   _h{};
    int   _levels{};
    float _w_grid_divider{};
    float _h_grid_divider{};
    int   _debug_octave_id{};

    Plane2D_float _data;
    Plane2D_float _intm;
    Plane2D_float _dog_3d;

public:
    // ... methods ...
};
```

**After (SYCL):**
```cpp
class Octave
{
    int   _w{};
    int   _h{};
    int   _levels{};
    float _w_grid_divider{};
    float _h_grid_divider{};
    int   _debug_octave_id{};

    Plane2D_float _data;
    Plane2D_float _intm;
    Plane2D_float _dog_3d;

    sycl::queue _queue;  // NEW: SYCL queue for this octave

public:
    // ... methods ...
};
```

**Purpose**: Each octave has its own SYCL queue for independent GPU operations.

---

##### 4.6.3 Added SYCL Queue Accessor Methods

**New methods added:**
```cpp
// NEW: SYCL queue access methods
inline sycl::queue& getQueue() { 
    return _queue; 
}

inline const sycl::queue& getQueue() const { 
    return _queue; 
}
```

**Purpose**: Provide access to the octave's SYCL queue for kernel submissions.

---

##### 4.6.4 Added Queue Initialization Method

**New method declaration:**
```cpp
/**
 * @brief Initialize SYCL queue for this octave
 */
void initQueue();
```

**Purpose**: Initialize the SYCL queue with GPU selector and properties.

---

#### 4.7 sift_octave.cc

**File**: `src/popsift/sift_octave.cc`

**Changes**: Implemented SYCL queue initialization and updated memory allocation.

##### 4.7.1 Added SYCL Header Include

**Change:**
```cpp
#include <sycl/sycl.hpp>  // NEW: Added for SYCL queue support
```

---

##### 4.7.2 Implemented initQueue() Method (NEW)

**New method implementation:**
```cpp
void Octave::initQueue()
{
    try {
        // Create queue with in-order and profiling properties
        auto props = sycl::property_list{
            sycl::property::queue::in_order(),
            sycl::property::queue::enable_profiling()
        };
        
        _queue = sycl::queue(sycl::gpu_selector_v, props);
        
        // Optional: Print device info for debugging
        auto device = _queue.get_device();
        std::cout << "Octave " << _debug_octave_id << " queue initialized on: " 
                  << device.get_info<sycl::info::device::name>() << std::endl;
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception in Octave::initQueue(): " << e.what() << std::endl;
        std::cerr << "Falling back to CPU for octave " << _debug_octave_id << std::endl;
        
        // Fallback to CPU with same properties
        auto props = sycl::property_list{
            sycl::property::queue::in_order(),
            sycl::property::queue::enable_profiling()
        };
        
        _queue = sycl::queue(sycl::cpu_selector_v, props);
    }
}
```

**Features**:
- Attempts GPU device selection with `sycl::gpu_selector_v`
- Creates in-order queue for sequential execution
- Enables profiling for performance analysis
- Falls back to CPU if GPU unavailable
- Exception handling for device initialization

---

##### 4.7.3 Updated alloc() Method

**Before (CPU):**
```cpp
void Octave::alloc( const Config& conf, int width, int height, int levels )
{
    _w = width;
    _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    _data  .alloc( width, height, levels );
    _intm  .alloc( width, height, levels );
    _dog_3d.alloc( width, height, levels-1 );
}
```

**After (SYCL):**
```cpp
void Octave::alloc( const Config& conf, int width, int height, int levels )
{
    _w = width;
    _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    // NEW: Initialize SYCL queue first
    initQueue();
    
    auto device = _queue.get_device();
    std::cout << "Octave " << _debug_octave_id 
              << " allocating on device: " 
              << device.get_info<sycl::info::device::name>() 
              << std::endl;

    // NEW: All planes now use this octave's queue
    _data  .alloc( width, height, levels, &_queue );
    _intm  .alloc( width, height, levels, &_queue );
    _dog_3d.alloc( width, height, levels-1, &_queue );
    
    // NEW: Verify allocation succeeded
    if (!_data.getDevicePtr() || !_dog_3d.getDevicePtr()) {
        throw std::runtime_error("Failed to allocate device memory for octave");
    }
    
    std::cout << "Octave " << _debug_octave_id 
              << " allocated: data=" << _data.getDevicePtr()
              << ", dog=" << _dog_3d.getDevicePtr()
              << std::endl;
}
```

**Key Changes**:
1. Call `initQueue()` before allocation
2. Pass `&_queue` to all `alloc()` calls
3. Added device info logging
4. Added allocation verification
5. Added debug output for device pointers

---

##### 4.7.4 Updated free() Method

**Before (CPU):**
```cpp
void Octave::free()
{
    _data  .dealloc();
    _intm  .dealloc();
    _dog_3d.dealloc();
}
```

**After (SYCL):**
```cpp
void Octave::free()
{
    // NEW: Wait for any pending operations on this queue before freeing
    try {
        _queue.wait();
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception during queue wait in Octave::free(): " 
                  << e.what() << std::endl;
    }

    // NEW: Use assignment to trigger smart pointer cleanup
    _data = PlaneD<float>();    
    _intm = PlaneD<float>();
    _dog_3d = PlaneD<float>();
}
```

**Key Changes**:
1. Added `_queue.wait()` to synchronize before freeing memory
2. Changed from explicit `dealloc()` to assignment (triggers destructors)
3. Added exception handling for queue synchronization

**Reason**: SYCL requires synchronization before freeing device memory to ensure no kernels are still using it.

---

#### 4.8 s_pyramid_build.cc

**File**: `src/popsift/s_pyramid_build.cc`

**Changes**: Converted CPU loops to SYCL parallel kernels for DoG (Difference of Gaussians) computation.

---

##### 4.8.1 Added SYCL Header Include

**Change:**
```cpp
#include <sycl/sycl.hpp>  // NEW: Added for SYCL queue support
```

**Location**: After existing includes (around line 20).

---

##### 4.8.2 Converted make_dog() from CPU to SYCL

**Before (CPU - nested loops):**
```cpp
static
void make_dog( PlaneD<float>& src,
               PlaneD<float>& dog,
               const int      w,
               const int      h,
               const int      max_level )
{
    for( int idy = 0; idy < h; idy++ )
    {
        for( int idx = 0; idx < w; idx++ )
        {
            float a = src.get( 0, idy, idx );
            for( int level = 0; level < max_level-1; level++ )
            {
                const float b = src.get( level+1, idy, idx );
                dog.set( level, idy, idx, b-a );
                a = b;
            }
        }
    }
}
```

**After (SYCL - parallel kernel):**
```cpp
static
sycl::event make_dog( sycl::queue&   q,
                      PlaneD<float>& src,
                      PlaneD<float>& dog,
                      const int      w,
                      const int      h,
                      const int      max_level )
{
    // Get device pointers 
    float* d_src = src.getDevicePtr();
    float* d_dog = dog.getDevicePtr();
    
    const int src_pitch = src.getPitch();
    const int dog_pitch = dog.getPitch();
    
    // Calculate grid dimensions (matching CUDA: 1024x1 blocks)
    const int block_x = 1024;
    const int block_y = 1;
    const int grid_x = (w + block_x - 1) / block_x;
    const int grid_y = (h + block_y - 1) / block_y;
    
    sycl::range<2> local_range(block_y, block_x);
    sycl::range<2> global_range(grid_y * block_y, grid_x * block_x);
    
    // Submit kernel and return event (non-blocking)
    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item) {
                const int idx = item.get_global_id(1);
                const int idy = item.get_global_id(0);
                
                if (idx < w && idy < h) {
                    const int pixel_offset = idy * src_pitch + idx;
                    float a = d_src[pixel_offset];
                    
                    for (int level = 0; level < max_level - 1; level++) {
                        const int next_level_offset = (level + 1) * h * src_pitch + pixel_offset;
                        const int dog_level_offset = level * h * dog_pitch + idy * dog_pitch + idx;
                        
                        const float b = d_src[next_level_offset];
                        d_dog[dog_level_offset] = b - a;
                        a = b;
                    }
                }
            });
    });
    
    // Return event without waiting - allows parallel execution
    return event;
}
```

**Key Changes:**
1. **Signature**: Added `sycl::queue&` parameter and `sycl::event` return type
2. **Device Pointers**: Use `getDevicePtr()` instead of `get()/set()` methods
3. **Pitch**: Use `getPitch()` for proper 2D memory indexing
4. **Parallelization**: Replace nested loops with `parallel_for` over 2D work-items
5. **Thread Indexing**: Use `item.get_global_id(0)` for y, `item.get_global_id(1)` for x
6. **Asynchronous**: Return `sycl::event` for non-blocking execution

---

##### 4.8.3 Updated dogs_from_blurred() Method

**Before (CPU - blocking call):**
```cpp
void Pyramid::dogs_from_blurred( int octave, int max_level )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    gauss::make_dog( oct_obj.getData(),
                     oct_obj.getDog(),
                     oct_obj.getWidth(),
                     oct_obj.getHeight(),
                     max_level );
}
```

**After (SYCL - returns event):**
```cpp
sycl::event Pyramid::dogs_from_blurred( int octave, int max_level )
{
    Octave& oct_obj = _octaves[octave];
    
    // Use the octave's own queue for parallel execution
    sycl::queue& q = oct_obj.getQueue();
    
    return gauss::make_dog( q,
                            oct_obj.getData(),
                            oct_obj.getDog(),
                            oct_obj.getWidth(),
                            oct_obj.getHeight(),
                            max_level);
}
```

**Key Changes:**
1. **Return type**: Changed from `void` to `sycl::event`
2. **Queue**: Get queue from octave with `oct_obj.getQueue()`
3. **Pass queue**: Pass queue reference to `make_dog()`
4. **Return event**: Return event from kernel submission

---

##### 4.8.4 Updated build_pyramid() for Parallel Execution

**Before (CPU - sequential processing):**
```cpp
void Pyramid::build_pyramid( const Config& conf, std::shared_ptr<ImageBase> base )
{
    // ... octave processing loop ...
    
    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave& oct_obj = _octaves[octave];
        POP_INFO2( conf.silent(), "call dogs_from_blurred" );
        dogs_from_blurred( octave, _levels );
    }
}
```

**After (SYCL - parallel execution with synchronization):**
```cpp
void Pyramid::build_pyramid( const Config& conf, std::shared_ptr<ImageBase> base )
{
    // ... octave processing loop ...
    
    // Launch DoG kernels asynchronously on all octaves
    // Each octave uses its own queue, so they execute in parallel
    std::vector<sycl::event> events;
    for( int octave=0; octave<_num_octaves; octave++ )
    {
        POP_INFO2( conf.silent(), "call dogs_from_blurred (async)" );
        
        // Submit kernel on octave's queue (non-blocking)
        sycl::event event = dogs_from_blurred( octave, _levels );
        events.push_back(event);
    }

    // Wait for all DoG computations to complete
    POP_INFO2( conf.silent(), "waiting for all DoG kernels to complete" );
    for (auto& e : events) {
        e.wait();
    }
    
    POP_INFO2( conf.silent(), "DoG computation complete" );
}
```

**Key Changes:**
1. **Event collection**: Store events from all octave submissions
2. **Parallel launch**: All octaves submit kernels without waiting
3. **Bulk synchronization**: Wait for all events at the end
4. **Improved logging**: Added async-specific messages

**Performance Benefit**: Multiple octaves can now process DoG in parallel on GPU.

---

##### 4.8.5 Thread Index Mapping (CPU to SYCL)

| CPU Loops | SYCL Equivalent | Description |
|-----------|-----------------|-------------|
| `for (int idy = 0; idy < h; idy++)` | `item.get_global_id(0)` | Y-axis iteration |
| `for (int idx = 0; idx < w; idx++)` | `item.get_global_id(1)` | X-axis iteration |
| Bounds check implicit | `if (idx < w && idy < h)` | Explicit bounds check |

---

##### 4.8.6 Memory Access Pattern Changes

**CPU Version (method calls):**
```cpp
float a = src.get( 0, idy, idx );                   // Level 0
const float b = src.get( level+1, idy, idx );      // Next level
dog.set( level, idy, idx, b-a );                   // Write result
```

**SYCL Version (pointer arithmetic with pitch):**
```cpp
const int pixel_offset = idy * src_pitch + idx;
float a = d_src[pixel_offset];                                        // Level 0

const int next_level_offset = (level + 1) * h * src_pitch + pixel_offset;
const float b = d_src[next_level_offset];                            // Next level

const int dog_level_offset = level * h * dog_pitch + idy * dog_pitch + idx;
d_dog[dog_level_offset] = b - a;                                     // Write result
```

**Why the change?**
- SYCL kernels access device memory directly via pointers
- Must respect pitch for proper 2D array layout
- Manual offset calculation for multi-level access

---

##  Known Issues and Limitations

### 1. OpenImageIO Incompatibility

**Issue**: OpenImageIO's bundled `fmt` library causes compilation errors with Intel DPC++ compiler.

**Error Message**:
```
/usr/include/OpenImageIO/detail/fmt/format.h:3122:38: error: 
implicit instantiation of undefined template 'fmt::detail::dragonbox::float_info<long double>'
```

**Root Cause**: The `fmt` library in OpenImageIO doesn't properly support `long double` with Intel compiler's type system.

**Workaround**: Disabled OpenImageIO. Use PGM format images only.

**Impact**: 
- Only PGM image format supported
- No support for JPEG, PNG, TIFF, etc.
- Must convert images to PGM before processing

**Solution**:
```bash
# Convert image to PGM
convert input.jpg output.pgm

# Run demo
./popsift-demo --input-file output.pgm
```

---

### 2. C++17 Requirement

**Issue**: SYCL 2020 requires minimum C++17 standard.

**Error Message**:
```
error: static assertion failed due to requirement '201402L >= 201703L': 
DPCPP does not support C++ version earlier than C++17.
static_assert(__cplusplus >= 201703L,
              ^~~~~~~~~~~~~~~~~~~~~~
```

**Solution**: Set C++17 in CMakeLists.txt (already implemented).

---

### 3. Compiler Requirement

**Issue**: Must use Intel DPC++ compiler (icpx). GCC/Clang will not work.

**Error Message** (if using GCC):
```
fatal error: sycl/sycl.hpp: No such file or directory
 #include <sycl/sycl.hpp>
          ^~~~~~~~~~~~~~~
```

**Solution**: Always specify compiler in CMake command:
```bash
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx
```

---

### 4. Intel oneAPI Installation Required

**Prerequisite**: Intel oneAPI Base Toolkit must be installed.

**Check Installation**:
```bash
which icpx
# Should output: /opt/intel/oneapi/compiler/latest/bin/icpx

icpx --version
# Should show: Intel(R) oneAPI DPC++/C++ Compiler
```

**Install if Missing**:
```bash
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# Or use APT (Ubuntu/Debian):
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-basekit
```

**Setup Environment**:
```bash
source /opt/intel/oneapi/setvars.sh
```

---

## Testing and Verification

### Build Instructions

```bash
# Clean build directory
cd ~/Desktop/popsift-SYCL/build
rm -rf *

# Configure with Intel DPC++ compiler
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DCMAKE_BUILD_TYPE=Release

# Expected output should include:
# -- ✓ Using Intel DPC++ compiler: /opt/intel/oneapi/compiler/2024.2/bin/icpx
# -- Build type: Release
# -- C++ Standard: 17
# -- CXX Flags:  -fsycl -std=c++17

# Build
make -j$(nproc)

# Expected output:
# [100%] Built target popsift
# [100%] Built target popsift-demo
# [100%] Built target popsift-match
```

### Run Tests

```bash
# Convert test image to PGM
convert sample.jpg sample.pgm

# Run demo
./Linux-x86_64/popsift-demo --input-file sample.pgm

# Expected output should show:
# - Device information (Intel GPU/CPU)
# - Number of features detected
# - Processing time
```

### Verify SYCL Device

```bash
# List available SYCL devices
sycl-ls

# Expected output (example):
# [opencl:gpu:0] Intel(R) Data Center GPU Max 1100
# [opencl:cpu:0] Intel(R) Xeon(R) CPU
# [level_zero:gpu:0] Intel(R) Arc(TM) A770 Graphics
```

## References

- **Intel oneAPI Documentation**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/documentation.html
- **SYCL 2020 Specification**: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html
- **Intel DPC++ Compiler**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html
- **CUDA to SYCL Migration Guide**: https://www.intel.com/content/www/us/en/developer/articles/guide/cuda-sycl-migration-guide.html
- **PopSift Original (CUDA)**: https://github.com/alicevision/popsift

---


**End of Migration Guide**