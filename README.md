<div width="100%" height="100%" align="center">
  
<h1 align="center">
  <p align="center">Professional CUDA C Programming</p>
  <a href="https://bcs.wiley.com/he-bcs/Books?action=index&itemId=1118739329&bcsId=9406">
    <img width="50%" src="cover.jpg" />
  </a>
</h1>
  

<b>John Cheng · Max Grossman · Ty McKercher 저</b></br>
Wrox(現 Willy) · 2014년 9월 9일 출시</br>
[[ERRATA](https://www.wiley.com/en-us/Professional+CUDA+C+Programming-p-9781118739327)] [[Code Samples](https://media.wiley.com/product_ancillary/29/11187393/DOWNLOAD/CodeSamples.zip)] [[Solutions](https://media.wiley.com/product_ancillary/29/11187393/DOWNLOAD/Solutions.zip)]</b> 

</div>

## :bulb: 목표

- **CUDA를 이용한 GPU 프로그래밍 기술을 익힌다.**

  > computer architecture와 parallel computing 개념을 익히고 CUDA를 이용해 실습한다.


</br>

## 🚩 정리한 문서 목록

### 📔 computer architecture, CUDA 기초

- [heterogeneous parallel computing with CUDA](https://github.com/erectbranch/CUDA_Basic/tree/master/ch01)

  > parallel computing, concurrency, sequential programming, task/data/block/cyclic parallelism

  > SISD/SIMD/MISD/MIMD, latency, bandwidth, throughput, multi-node with distributed memory/multiprocessor with shared memory, heterogeneous computing, host/device code

  > CUDA 간단 예제: hello world 출력

- [CUDA Programming Model](https://github.com/erectbranch/CUDA_Basic/tree/master/ch02)

  > CUDA programming model, unified memory, asynchronous

  > CUDA를 통한 managing memory(cudaMalloc, cudaMemcpy, cudaMemset, cudaFree), organizing thread(grid, block 정의하는 방법), launching kernel

  > (macro를 이용한) handling error, timing kernel(CPU timer/nvprof 이용), nvprof를 이용한 ratio of instruction:bytes 계산

  > vector addition(+ host/device side 연산 결과 비교), matrix addition(2D grid, 2D blocks/1D grid, 1D blocks/ 2D grid, 1D blocks)

<br/>

## :mag: 목차

### Chapter 1: Heterogeneous Parallel Computing with CUDA 1

    Parallel Computing 2

    Sequential and Parallel Programming 3

    Parallelism 4

    Computer Architecture 6

    Heterogeneous Computing 8

    Heterogeneous Architecture 9

    Paradigm of Heterogeneous Computing 12

    CUDA: A Platform for Heterogeneous Computing 14

    Hello World from GPU 17

    Is CUDA C Programming Difficult? 20

### Chapter 2: CUDA Programming Model 23

    Introducing the CUDA Programming Model 23

    CUDA Programming Structure 25

    Managing Memory 26

    Organizing Threads 30

    Launching a CUDA Kernel 36

    Writing Your Kernel 37

    Verifying Your Kernel 39

    Handling Errors 40

    Compiling and Executing 40

    Timing Your Kernel 43

    Timing with CPU Timer 44

    Timing with nvprof 47

    Organizing Parallel Threads 49

    Indexing Matrices with Blocks and Threads 49

    Summing Matrices with a 2D Grid and 2D Blocks 53

    Summing Matrices with a 1D Grid and 1D Blocks 57

    Summing Matrices with a 2D Grid and 1D Blocks 58

    Managing Devices 60

    Using the Runtime API to Query GPU Information 61

    Determining the Best GPU 63

    Using nvidia-smi to Query GPU Information 63

    Setting Devices at Runtime 64

### Chapter 3: CUDA Execution Model 67

    Introducing the CUDA Execution Model 67

    GPU Architecture Overview 68

    The Fermi Architecture 71

    The Kepler Architecture 73

    Profile-Driven Optimization 78

    Understanding the Nature of Warp Execution 80

    Warps and Thread Blocks 80

    Warp Divergence 82

    Resource Partitioning 87

    Latency Hiding 90

    Occupancy 93

    Synchronization 97

    Scalability 98

    Exposing Parallelism 98

    Checking Active Warps with nvprof 100

    Checking Memory Operations with nvprof 100

    Exposing More Parallelism 101

    Avoiding Branch Divergence 104

    The Parallel Reduction Problem 104

    Divergence in Parallel Reduction 106

    Improving Divergence in Parallel Reduction 110

    Reducing with Interleaved Pairs 112

    Unrolling Loops 114

    Reducing with Unrolling 115

    Reducing with Unrolled Warps 117

    Reducing with Complete Unrolling 119

    Reducing with Template Functions 120

    Dynamic Parallelism 122

    Nested Execution 123

    Nested Hello World on the GPU 124

    Nested Reduction 128


### Chapter 4: Global Memory 135

    Introducing the CUDA Memory Model 136

    Benefits of a Memory Hierarchy 136

    CUDA Memory Model 137

    Memory Management 145

    Memory Allocation and Deallocation 146

    Memory Transfer 146

    Pinned Memory 148

    Zero-Copy Memory 150

    Unified Virtual Addressing 156

    Unified Memory 157

    Memory Access Patterns 158

    Aligned and Coalesced Access 158

    Global Memory Reads 160

    Global Memory Writes 169

    Array of Structures versus Structure of Arrays 171

    Performance Tuning 176

    What Bandwidth Can a Kernel Achieve? 179

    Memory Bandwidth 179

    Matrix Transpose Problem 180

    Matrix Addition with Unified Memory 195


### Chapter 5: Shared Memory and Constant Memory 203

    Introducing CUDA Shared Memory 204

    Shared Memory 204

    Shared Memory Allocation 206

    Shared Memory Banks and Access Mode 206

    Configuring the Amount of Shared Memory 212

    Synchronization 214

    Checking the Data Layout of Shared Memory 216

    Square Shared Memory 217

    Rectangular Shared Memory 225

    Reducing Global Memory Access 232

    Parallel Reduction with Shared Memory 232

    Parallel Reduction with Unrolling 236

    Parallel Reduction with Dynamic Shared Memory 238

    Effective Bandwidth 239

    Coalescing Global Memory Accesses 239

    Baseline Transpose Kernel 240

    Matrix Transpose with Shared Memory 241

    Matrix Transpose with Padded Shared Memory 245

    Matrix Transpose with Unrolling 246

    Exposing More Parallelism 249

    Constant Memory 250

    Implementing a 1D Stencil with Constant Memory 250

    Comparing with the Read-Only Cache 253

    The Warp Shuffle Instruction 255

    Variants of the Warp Shuffle Instruction 256

    Sharing Data within a Warp 258

    Parallel Reduction Using the Warp Shuffle Instruction 262


### Chapter 6: Streams and Concurrency 267

    Introducing Streams and Events 268

    CUDA Streams 269

    Stream Scheduling 271

    Stream Priorities 273

    CUDA Events 273

    Stream Synchronization 275

    Concurrent Kernel Execution 279

    Concurrent Kernels in Non-NULL Streams 279

    False Dependencies on Fermi GPUs 281

    Dispatching Operations with OpenMP 283

    Adjusting Stream Behavior Using Environment Variables 284

    Concurrency-Limiting GPU Resources 286

    Blocking Behavior of the Default Stream 287

    Creating Inter-Stream Dependencies 288

    Overlapping Kernel Execution and Data Transfer 289

    Overlap Using Depth-First Scheduling 289

    Overlap Using Breadth-First Scheduling 293

    Overlapping GPU and CPU Execution 294

    Stream Callbacks 295


### Chapter 7: Tuning Instruction-Level Primitives 299

    Introducing CUDA Instructions 300

    Floating-Point Instructions 301

    Intrinsic and Standard Functions 303

    Atomic Instructions 304

    Optimizing Instructions for Your Application 306

    Single-Precision vs. Double-Precision 306

    Standard vs. Intrinsic Functions 309

    Understanding Atomic Instructions 315

    Bringing It All Together 322


### Chapter 8: GPU-Accelerated CUDA Libraries and OpenACC 327

    Introducing the CUDA Libraries 328

    Supported Domains for CUDA Libraries 329

    A Common Library Workflow 330

    The CUSPARSE Library 332

    cuSPARSE Data Storage Formats 333

    Formatting Conversion with cuSPARSE 337

    Demonstrating cuSPARSE 338

    Important Topics in cuSPARSE Development 340


    The cuBLAS Library 341

    Managing cuBLAS Data 342

    Demonstrating cuBLAS 343

    Important Topics in cuBLAS Development 345


    The cuFFT Library 346

    Using the cuFFT API 347

    Demonstrating cuFFT 348


    The cuRAND Library 349

    Choosing Pseudo- or Quasi- Random Numbers 349

    Overview of the cuRAND Library 350

    Demonstrating cuRAND 354

    Important Topics in cuRAND Development 357

    CUDA Library Features Introduced in CUDA 6 358

    Drop-In CUDA Libraries 358

    Multi-GPU Libraries 359

    A Survey of CUDA Library Performance 361

    cuSPARSE versus MKL 361

    cuBLAS versus MKL BLAS 362

    cuFFT versus FFTW versus MKL 363


    Using OpenACC 365

    Using OpenACC Compute Directives 367

    Using OpenACC Data Directives 375

    The OpenACC Runtime API 380

    Combining OpenACC and the CUDA Libraries 382
    OpenACC 384


### Chapter 9: Multi-GPU Programming 387

    Moving to Multiple GPUs 388

    Executing on Multiple GPUs 389

    Peer-to-Peer Communication 391

    Synchronizing across Multi-GPUs 392

    Subdividing Computation across Multiple GPUs 393

    Allocating Memory on Multiple Devices 393

    Distributing Work from a Single Host Thread 394

    Compiling and Executing 395

    Peer-to-Peer Communication on Multiple GPUs 396

    Enabling Peer-to-Peer Access 396

    Peer-to-Peer Memory Copy 396

    Peer-to-Peer Memory Access with Unified Virtual Addressing 398

    Finite Difference on Multi-GPU 400

    Stencil Calculation for 2D Wave Equation 400

    Typical Patterns for Multi-GPU Programs 401

    2D Stencil Computation with Multiple GPUs 403

    Overlapping Computation and Communication 405

    Compiling and Executing 406

    Scaling Applications across GPU Clusters 409

    CPU-to-CPU Data Transfer 410

    GPU-to-GPU Data Transfer Using Traditional MPI 413

    GPU-to-GPU Data Transfer with CUDA-aware MPI 416

    Intra-Node GPU-to-GPU Data Transfer with CUDA-Aware MPI 417

    Adjusting Message Chunk Size 418

    GPU to GPU Data Transfer with GPUDirect RDMA 419


### Chapter 10: Implementation Considerations 425

    The CUDA C Development Process 426

    APOD Development Cycle 426

    Optimization Opportunities 429

    CUDA Code Compilation 432

    CUDA Error Handling 437

    Profile-Driven Optimization 438

    Finding Optimization Opportunities Using nvprof 439

    Guiding Optimization Using nvvp 443

    NVIDIA Tools Extension 446

    CUDA Debugging 448

    Kernel Debugging 448

    Memory Debugging 456


    A Case Study in Porting C Programs to CUDA C 462

    Assessing crypt 463

    Parallelizing crypt 464

    Optimizing crypt 465

    Deploying Crypt 472
    Porting crypt 475
