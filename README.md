# lab-nbody
N-Body CUDA simulation - Simple all-pairs N-Body algorithm.

## Getting started

### Build

``` bash
git clone https://github.com/gcoe-dresden/lab-nbody.git
cd lab-nbody
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Run

``` bash
./nbody [nbodies] [device-index]
# Runs nbody with 8192 nbodies
./nbody 8192
```
The current version includes validation of the results on the CPU.
However, due to the serial performance, validation is restricted to <8193.

### Tasks

- Analyze the code for possible performance improvements.
- Choose the most promising optimization and implement it.
- Measure the speedup.
- Repeat the process.

### Questions

- What is the maximal performance?
- The measured performance can be higher than the maximal performance. Why?
- Does it scale?
- How is the performance for different blocksizes?
- What is the maximum blocksize?

### Profiling CUDA kernels - Current Issues

With CUDA driver 418.43+ admin privileges are required to gather the metrics/events from the device.
With older CUDA versions < 10.2, errors look a bit different, e.g. `Error: Internal profiling error 4183:7.`
As of CUDA driver version 418.43 privileges for root/CAP_SYS are required, to use nvprof CUPTI, see:

     https://docs.nvidia.com/cupti/Cupti/r_overview.html#r_whats_new
