# 1 Heterogeneous Parallel Computing with CUDA

**high-performance computing**(HPC)를 실현하기 위해서는 computer architecture뿐만 아니라, hardware systems, software tool, programming platforms, 그리고 parallel programming paradigms를 고려해야 한다.

> 들어가기 앞서 2014년 출된 교재는 CUDA 6.0와 Kepler architecture를 사용하므로, 현재 실정(CUDA 12.0.x, Ada Lovelace architecture인 RTX 4090)과는 거리가 있다.

---

## 1.1 Parallel Computing

parallel computing이란 여러 연산을 simultaneous하게 수행하고, 큰 problem을 작은 problem들로 나눠서 **concurrent**하게 수행하는 것을 의미한다. 이때 이렇게 작게 나뉜 연산 piece를 **task**라고 한다.

여기서 **concurrency**(동시성)이란 정확히는 동시에 실행되는 것'처럼' 보이는 개념이다. 예를 들어 한 개의 core에서도 작업들을 **context switch**하는 방법으로 여러 작업을 concurrent하게 수행할 수 있다.

![context switch](images/context_switch.png)

> 이 경우 concurrent하게 실행되는 task들은 OS의 scheduling 정책에 의해 서로 복잡하게 얽힌다.

좀 더 프로그래머 관점에서 보면 concurrent한 연산을 어떻게 구현할지가 목표가 된다. 크게 두 가지 관점으로 구현할 수 있다.

- Computer architecture(하드웨어 관점)

- Parellel programming(소프트웨어 관점)

computer architecture 관점에서는 problem을 multiple processes 또는 multiple thread을 이용해서 concurrent하게 해결하고자 한다. 여기서 제일 중요한 component가 주로 **core**라고 지칭하는 **CPU**(Central Processing Unit)이다.

현재는 hardware architecture 수준에서 parallelism을 지원하는 multicore가 존재한다. 여기서 programming은 연산이 가능한 core들에게 효율적으로 mapping이 되도록 하는 역할을 맡는 것이다.

> 특히 multicore에서의 algorithm 구현을 위해서는, computer architecture의 구조를 상세하게 이해할 필요가 있다.

---

## 1.2 Sequential and Parallel Programming

- sequential programming

  ![sequential programming](images/sequential_programming.png)

- parallel programming

  ![sequential parallel programming](images/sequential_parallel_programming.png)

program은 기본적으로 **instruction**와 **data** 두 개의 ingredients를 갖는다. 

- instruction은 input을 받아서 function을 수행(ALU 등이 연산)하고 output을 도출한다. 

- 이때 instruction 처리 중에 발생할 수 있는 **data dependency**를 주의해야 한다.

---

## 1.3 Parallelism

현재 **parallelism**(병렬성)은 크게 두 가지 type으로 분류할 수 있다.

- Task parallelism

  **Task parallelism**은 많은 task나 function들을 independent하게 수행할 수 있을 때 수행한다. task parallelism은 multiple cores에 function들이 분배되게 만든다.

- Data parallelism

  data element들을 parallel threads로 mapping한다. 따라서 굉장히 큰 data set을 처리하는 여러 application에서 speedup을 얻는 데 유용하다. 
  
  > CUDA programming이 바로 data parallelism에 특화된 방법이다.

---

### 1.3.1 data parallelism

data parallel program을 design하기 위해서는, 우선 data를 여러 thread로 **partition**(분할)해야 한다.

- block partitioning

  **block partitioning**은 여러 element를 **chunk**로 묶인다. 
  
  - (대체로) single thread가 한 번에 한 chunk를 처리한다.

  아래 그림이 1차원 data를 partition한 block partition이다.

  ![block partition](images/block_partition_ex.png)

- cyclic partitioning

  **cyclic partitioning**에서는 좀 더 적은 양의 data element를 chunk로 묶는다. 
  
  - 각 thread가 하나 이상의 chunk를 처리한다.

  아래 그림이 1차원 data를 partition한 cyclic partition이다.

  ![cyclic partition](images/cyclic_partition_ex.png)

<br/>

2차원 이상의 data일 때는 어떨까? 2차원의 경우 다음과 같이 x,y축을 기준으로 partitioning할 수 있다. 

![partition on dimenstion](images/partition_on_dimension.png)

> 사실 hardware적인 관점에서는 data가 physical storage에 결국 1차원으로 mapping된다. 

---

## 1.4 Computer Architecture

대체로 아래 **Flynn's Taxonomy**를 이용해 computer architecture를 분류하는 경우가 많다. 이 분류에서는 computer architecture를 4가지 type으로 나눈다.

![Flynn's Taxonomy](images/computer_architecture_classification.png)

- **SISD**(Single Instruction Single Data)

  (traditional computer) 오직 core가 한 개 존재하며, 한 번에 오직 한 instruction stream만 실행된다. 

- **SIMD**(Single Instruction Multiple Data)

  (대부분의 modern computer) parallel architecture에 속한다. multiple core를 가지며 '모든 core가 한 번에 같은 instruction을 execute'한다. 
  
  - SIMD의 가장 큰 장점: 프로그래머들이 sequential하게 program을 구성해도, compiler 덕분에 parallel speedup을 얻을 수 있다.

- **MISD**(Multiple Instruction Single Data)

  비교적 드문 architecture로, multiple core가 seperate instruction stream로 같은 data stream을 처리한다.

- **MIMD**(Multiple Instruction Multiple Data)

  multiple core를 사용한 parallel architecture다. multiple data stream으로 구성되며 각각 independent instruction을 처리한다. 
  
  > 참고로 대부분의 MIMD architecture는 sub-component로 SIMD를 포함한다.

---

### 1.4.1 performance

architecture level에서 다음 지표를 주로 performance로 사용한다. 따라서 다음 지표들을 높일 수 있는 방법을 고려해야 한다.

- **latency**: operation이 시작해서 끝날 때까지의 시간(주로 ms 단위 사용)

  > response time, 혹은 execution time으로도 부른다.

- **bandwidth**: 대역폭. 단위 시간당 처리하는 data의 양.(주로 megabytes/sec 혹은 gigabytes/sec 단위 사용)

  - gflops: billion floating-point operations per second를 의미한다.

- **throughput**: 처리율. 단위 시간당 처리하는 operation의 양.(주로 gflops 단위 사용)

개개인의 컴퓨터 사용자 입장에서는 주로 하루에 얼마나 더 많은 일을 할 수 있는지가 중요하기 때문에, latency를 더 중요하게 볼 가능성이 높다. 하지만 여러 사용자가 사용하는 데이터센터 입장에서는, 얼마나 더 많은 일을 처리할 수 있는가가 더 중요하므로 throughput을 더 중요하게 볼 가능성이 높다.

---

### 1.4.2 memory organization

computer architecture는 memory organization 관점에서도 크게 두 가지로 분류할 수 있다.

- Multi-node with **distributed** memory

  multi-node system에서는 networking을 통해, 여러 processor가 통신하며 연산을 수행한다.

  ![clusters](images/clusters.png)

  각 processor는 자신만의 local memory를 가지며, networking으로 content를 communicate한다. 이런 systems를 주로 **clusters**라고 지칭한다.

- Multiprocessor with **shared** memory

  processor들이 보통 물리적으로 같은 memory와 연결되어 있거나, PCIe(PCI-Express)와 같은 low-latency link를 통해 data를 share한다.

  > multiprocessor는 dual-processor부터 수백 개가 넘는 processor까지 다양한 size를 지칭한다.

  ![multiprocessor](images/multiprocessor.png)

  shared address space를 사용하지만, memory sharing을 위해서 꼭 물리적으로 연결된 단일 memory가 필요하지는 않다. 

    
> multicore architecture 중에서도 수십, 수백 개의 core를 가진 것을 **many-core**라고 지칭한다. GPU가 many-core architecture에 해당된다.

> GPU는 모든 종류의 parallelism을 가지고 있다. multithreading, MIMD, SIMD, instruction-level parallelism.

---

## 1.5 heterogeneous computing

> homogeneous system에서 heterogeneous system으로의 전환을 HPC로 가는 기념비적인 사건으로 보는 시점도 있다.

보통 CPU와 GPU는 single computer node에서 PCIe bus로 연결되는 discrete한 방식으로 구성된다. 이런 system이 바로 **heterogeneous system**이며 각자가 well-suited task를 맡아 수행한다. 

하지만 이런 heterogeneous system도 design complexity에 의해 여러 limit를 가지게 된다.

---

### 1.5.1 heterogeneous architecture

2014년 기준으로 heterogeneous computer node는 대체로 두 개의 multicore CPU sockets와 many-core GPU로 구성되었다. 이때 GPU는 standalone하지 않고 CPU의 co-processor 역할을 한다. 따라서 CPU를 **host**, GPU를 **device**라고 지칭한다.

![heterogeneous architecture](images/heterogeneous_architecture.png)

heterogeneous application은 두 가지 종류의 code로 구성된다.(application 자체는 보통 CPU에서 initialize한다.)

- CPU에서 구동되는 host code

  host code는 compute-intensive(많은 계산이 필요한) task를 load하기 전에 environment, code, data를 관리한다.

  > CPU는 control-intensive task에 적합하다.

- GPU에서 구동되는 device code

  computational intensive application의 경우, program은 보통 많은 양의 data parallelism을 가진다. GPU를 이 data parallelism 처리를 accelerate하기 위해 사용한다.

이처럼 CPU와 물리적으로 떨어져서 application의 computationally intensive section을 accelerate하기 위해 사용하는 hardware component를 **hardware accelerator**라고 한다. GPU가 대표적이다.

> 대표적인 GPU 라인업으로 Tegra(mobile, embedded device), GeForce(consumer graphics), Quadro(professional visualization), Tesla(datacenter parallel computing)이 있다.

> GPU capability에 영향을 미치는 가장 큰 요소로 <U>CUDA core의 개수, memory size</U>를 꼽을 수 있다. 또한 GPU performance를 나타내는 방법으로는 주로 <U>Peak computational performance, memory bandwidth</U>를 사용한다.

---

### 1.5.2 paradigm of heterogeneous computing

CPU와 GPU가 어떤 task에서 유용한지 알기 위해 parallelism level과 data size를 기준으로 비교해 보자.

![CPU, GPU](images/CPU_and_GPU.png)

![code portion](images/heterogeneous_parallel_computing.png)

- CPU: 작은 data size, 정교한 control logic, low-level parallelism에 유용하다. 또한 복잡한 logic과 instruction-level parallelism 처리에 적합하다.

- GPU: 많은 양의 data이며 굉장히 많은 data parallelism을 가진 경우에 유용하다. 게다가 CPU보다 큰 bandwidth를 가진다.

---

## 1.6 CUDA

![CUDA platform](images/cuda_platform.png)

**CUDA**(Compute Unified Device Architecture)는 NVIDIA GPU에서 복잡한 계산 문제를 더 효율적이게 수행하기 위해 만들어진 parallel computing platform이자 programming model이다.

CUDA platform은 CUDA-accelerated library, compiler directives, application programming interfaces, 그리고 다른 프로그래밍 언어(C, C++, Fortran, Python)의 extension 등으로 사용할 수 있다.

CUDA C 역시 표준 ANSI C의 extension이다. CUDA를 사용하면 core의 수를 조절하는 방식으로 GPU의 parallelism을 조절할 수 있다.

CUDA는 GPU device와 thread를 관리하기 위해 두 API level을 제공한다.

![CUDA API](images/CUDA_API_level.png)

- CUDA Driver API: low-level API로 program과 관련이 깊다. 그러나 GPU 동작 원리에 특화되어 있다. 

- CUDA Runtime API: driver API상에서 구현된 high-level API로, runtime API의 각 function들이 basic operation으로 분해되서 driver API로 넘겨진다.

CUDA program도 host code(CPU에서 동작), device code(GPU에서 동작)으로 나뉜다. host code를 compile하는 과정에서 CUDA nvcc compiler가 device code를 분리해 준다.

![CUDA host, device code](images/CUDA_host_device_code.png)

- host code: 표준 C code이며, C compiler가 compile한다.

- device code: GPU에서 실행되는 **kernel**(data-parallel function)을 가진 CUDA C code이다. nvcc가 compile한다.

> nvcc는 LLVM compiler 기반으로 만들어졌다.

---

## 1.6.1 hello world from GPU

GPU에서 kernel code를 작성하여 "Hello World." 문자열을 출력할 것이다.

우선 CUDA compiler가 설치되어 있는지 확인한다.(경로는 보통 /usr/local/cuda/bin/nvcc)

```bash
$ which nvcc
```

![which nvcc](images/which_nvcc.png)

확인이 끝났다면 장치에 GPU accelerator card가 장착되었는지 확인한다.

```bash
$ ls -l /dev/nv*
```

![GPU accelerator card](images/GPU_accelerator_card.png)

> nvidia0이 등록된 GPU card이다.(card가 더 있다면 nvidia1, nvidia2식으로 더 있을 것이다.)

확인이 되었다면 CUDA C code를 작성한다. 확장자는 .cu로 하며 CUDA nvcc compiler를 이용해서 program을 compile하도록 설정한다. kernel code를 포함하여 작성하고 compile한 뒤, 명령을 입력해서 executable file을 실행하면 된다.

우선 비교를 위해 C 언어로만 작성한 hello world 출력 코드를 보자. 파일은 'hello.cu'란 이름으로 저장한다.

```c
#include <stdio.h>
int main(void){
  printf("Hello World from CPU!\n");
}
```

다음 명령을 통해 nvcc로 compile한다.

```bash
nvcc hello.cu -o hello
```

그 다음으로는 kernel function을 이용해 본다. 파일명은 'helloFromGPU.cu'이다.

```c
#include <stdio.h>

__global__ void helloFromGPU(void){
  printf("Hello World from GPU!\n");
}
 
int main(void){
  // hello from cpu
  // printf("Hello World from CPU!\n");

  helloFromGPU <<<1, 10>>>();
  cudaDeviceReset();
  return 0;
}
```

- \_\_global\_\_: compiler에게 function이 CPU에서 call되어, GPU에서 execute됨을 알린다.

- helloFromGPU <<<1,10>>>();: kernel function을 실행한다.(host thread에서 call되어 device side에서 실행된다.) 다수의 thread array에서 같은 code를 실행한다. 

  - parameter는 10개의 GPU thread로 구성된 block 한 개를 만든다는 뜻이다. 모든 thread는 같은 code를 실행한다.

- cudaDeviceReset(): 현재 process와 관련된 모든 resource들을 device에서 clean up(destroy)한다.

만들어진 파일을 ./helloFromGPU로 실행한 결과는 다음과 같다.

![helloFromGPU 실행 결과](images/helloFromGPU_result.png)

CUDA program 구성을 간단히 요약하면 다음과 같은 과정으로 전개된다.

1. GPU memory들을 allocate.

2. CPU memory에서 GPU memory로 data를 copy.

3. CUDA kernel을 invoke.

4. GPU memory에서 CPU memory로 data를 copy back한다.

5. GPU memory를 destroy.

> 간단한 helloFromGPU.cu 예제에서는 오직 3가지 과정만 있었다.

---

### 1.6.2 CUDA C programming 필요성

parallel하며 효과적인 code를 작성하기 위해서는 CPU architecture를 알아야 한다. 예를 들면 **locality**(지역성)와 같은 매우 중요한 개념을 모르면 불가능하다. locality란 data를 reuse(재사용)하여 memory access latency를 줄이는 방법을 의미한다.

locality는 두 가지 종류로 나눌 수 있다.

- **Temporal locality**: 짧은 기간동안 존재하는 data나 resource를 재사용.

- **Spatial locality**: storage location에서 가깝게 위치한 data elements의 재사용.

현재 CPU architecture는 이 두 locality를 활용하기 위해 큰 cache들을 사용한다. 또한 이런 CPU cache를 효율적으로 사용하려면 프로그래머가 algorithm을 제대로 디자인하는 것이 중요하다.

> 예를 들어 main memory의 speedup을 위해 사용되는 software-managed cache인 **shared memory**가 있다. shard memory로 code의 locality를 control할 수 있다.

다시 말해 프로그래머는 low-level cache optimization이 가능해야 한다. 하지만 CPU가 어떻게 schedule되는지 프로그래머가 알기는 어렵다.

이때 CUDA를 memory hierarchy와 thread hierarchy를 확인할 수 있는 유용한 수단으로 사용할 수 있다.

---



