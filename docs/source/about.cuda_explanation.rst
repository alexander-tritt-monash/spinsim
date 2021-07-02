****************************************
Crash course of numba cuda for beginners
****************************************

This package is written for the possible use and modification by those who have not encountered GPU based code before, so here is a brief introduction.

GPU functions that are called from a CPU, are called *kernels*.
*Numba* is a *Just In Time* (JIT) compiler that can compile python code into kernels.
GPUs as a whole are also known as *streaming multiprocessors* (SMs).
Also, the CPU is sometimes called the *host*, with the GPU it controls called the *device*.

An instance of a kernel program running on a GPU is called a *thread*.
There can be thousands of threads of the same kernel running on a SM at once, each running the same code.
Each thread is given a unique index, which lets it know which part of a larger array it should work on (you decide what that array is).
Groups of 32 SM hardware *core* run threads in sync with each other.
These groups are called *warps*.
All threads in a warp are finished executing before they are switched out for more threads.
In software, threads with similar indices are organised into groups called *blocks*.
Warps are filled with threads of the same block.
The array of all blocks to be executed is called a *grid*.

When executing a kernel, you must specify the size of blocks (in terms of threads) and the size of the grid (in terms of blocks).
In numba, these are written in square brackets after the function name and before the function arguments.

.. code-block:: python

   kernelName[blocksPerGrid, threadsPerBlock](arguments)

If the size of the array to be worked on does not fit in nicely with the block and grid sizes, then you need to check whether the kernel index is accessing somewhere outside the array boundary or not, or else other memory might be overwritten.

Kernels cannot return values, but any arrays that are passed in as arguments can be modified by the kernel.

Each SM has a set number of *registers* (fast memory) that can be shared between all cores.
The number of occupied cores can be increased by limiting the number of registers used per thread with the max_registers option in the cuda.jit decorator.
However, Doing so too much may also slow down code since it will be using slower memory more often.
Threads each have their own *local memory* that only they can access, and *shared memory* that is available to all other threads.
The arguments to the kernel are all shared.

*Device functions* can also be written.
These are functions that are executed within a kernel thread (ie on the device), and cannot be called by the CPU (ie the host).

Numba can also be used to write JIT compiled code for the CPU.
prange can be used as an alternative to range in for loops to be able to take advantage of CPU parallel processing such as multiple cores, or vector coprocessors.
