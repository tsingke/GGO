#include "ggo.h"

//算法应用名为getfit的变量名字，用前先改名
__device__ static float getfit(const float* x) {//testf1
    float sum = 0;
    for (int i = 0; i < dimension; i++)
        sum += (x[i] - i) * (x[i] - i);
    return sum;
}

__host__ static float h_getfit(const float x[]) {//h_testf1
    float sum = 0;
    for (int i = 0; i < dimension; i++)
        sum += (x[i] - i) * (x[i] - i);
    return sum;
}


// F1: Sphere's Problem
__device__ float f1(const float* x) {
    float sum = 0;
    for (int i = 0; i < dimension; i++)
        sum += (x[i] - i) * (x[i] - i);
    return sum;
}


// F2: Schwefel's Problem
__device__ float f2(float* x) {
    float max_val = fabsf(x[0]);
    for (int i = 1; i < dimension; ++i) {
        if (fabsf(x[i]) > max_val) {
            max_val = fabsf(x[i]);
        }
    }
    return max_val;
}

// F3: Step Function
__device__ float f3(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += (floor(x[i] + 0.5)) * (floor(x[i] + 0.5));
    }
    return sum;
}

// F4: Ackley's Function
__device__ float f4(float* x) {
    float sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum1 += x[i] * x[i];
        sum2 += cosf(2 * PI * x[i]);
    }
    float a = 20, b = 0.2, c = 2 * PI;
    return -a * expf(-b * sqrtf(sum1 / dimension)) - expf(sum2 / dimension) + a + expf(1);
}

// F5: Rastrigin's Function
__device__ float f5(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i] - 10 * cosf(2 * PI * x[i]);
    }
    return 10 * dimension + sum;
}

// F6: Griewank's Function
__device__ float f6(float* x) {
    float sum = 0.0;
    float prod = 1.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i];
        prod *= cosf(x[i] / sqrtf(i + 1));
    }
    return sum / 4000 - prod + 1;
}

// F7: Sphere Function
__device__ float f7(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

// 测试函数接口
__device__ float benchmark_func(float* x, int func_num) {
    switch (func_num) {
    case 1: return f1(x);
    case 2: return f2(x);
    case 3: return f3(x);
    case 4: return f4(x);
    case 5: return f5(x);
    case 6: return f6(x);
    case 7: return f7(x);

    default: return -1.0; // Invalid function number
    }
}

//cpu版本

// F1: Sphere's Problem
float cpu_f1(const float* x) {
    float sum = 0;
    for (int i = 0; i < dimension; i++)
        sum += (x[i] - i) * (x[i] - i);
    return sum;
}

// F2: Schwefel's Problem 2.21
float cpu_f2(float* x) {
    float max_val = fabsf(x[0]);
    for (int i = 1; i < dimension; ++i) {
        if (fabsf(x[i]) > max_val) {
            max_val = fabsf(x[i]);
        }
    }
    return max_val;
}

// F3: Step Function
float cpu_f3(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += (floor(x[i] + 0.5)) * (floor(x[i] + 0.5));
    }
    return sum;
}

// F4: Ackley's Function
float cpu_f4(float* x) {
    float sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum1 += x[i] * x[i];
        sum2 += cosf(2 * PI * x[i]);
    }
    float a = 20, b = 0.2, c = 2 * PI;
    return -a * expf(-b * sqrtf(sum1 / dimension)) - expf(sum2 / dimension) + a + expf(1);
}

// F5: Rastrigin's Function
float cpu_f5(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i] - 10 * cosf(2 * PI * x[i]);
    }
    return 10 * dimension + sum;
}

// F6: Griewank's Function
float cpu_f6(float* x) {
    float sum = 0.0;
    float prod = 1.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i];
        prod *= cosf(x[i] / sqrtf(i + 1));
    }
    return sum / 4000 - prod + 1;
}

// F7: Sphere Function
float cpu_f7(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

// 测试函数接口
float cpu_benchmark_func(float* x, int func_num) {
    switch (func_num) {
    case 1: return cpu_f1(x);
    case 2: return cpu_f2(x);
    case 3: return cpu_f3(x);
    case 4: return cpu_f4(x);
    case 5: return cpu_f5(x);
    case 6: return cpu_f6(x);
    case 7: return cpu_f7(x);

    default: return -1.0; // Invalid function number
    }
}

//getnorm()函数的实现
__device__ static float go_norm(float* a)
{
    float c = 0;
    for (int i = 0; i < dimension; ++i)
        c += a[i] * a[i];
    return sqrtf(c);
}

__device__ static float getrand(int idx, float min, float max, curandState* states) {
    return min + (max - min) * curand_uniform(&states[idx]);
}

//初始化核函数
__global__ void init_particle(float* positions, float* fitness, curandState* states, unsigned long seed, const int func_num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < popsize * dimension) {
        curand_init(seed, threadIdx.x, 0, &states[threadIdx.x]);
        positions[idx] = curand_uniform(&states[threadIdx.x]) * (float)dimension;
        __syncthreads();
        if (threadIdx.x == 0)
            fitness[blockIdx.x] = benchmark_func(&positions[idx], func_num);
    }
}

// Device function to merge two sorted subarrays
__device__ static void merge(float* fitness, float* temp, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (fitness[i] <= fitness[j]) {
            temp[k++] = fitness[i++];
        }
        else {
            temp[k++] = fitness[j++];
        }
    }
    while (i <= mid) {
        temp[k++] = fitness[i++];
    }
    while (j <= right) {
        temp[k++] = fitness[j++];
    }

    for (i = left; i <= right; i++) {
        fitness[i] = temp[i];
    }
}

// CUDA kernel for merge sort at a single level
__global__ void mergeSortKernel(float* fitness, float* temp, int width, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * width * 2;

    if (left + width < n) {
        int mid = left + width - 1;
        int right = min(left + 2 * width - 1, n - 1);
        merge(fitness, temp, left, mid, right);
    }
}
__global__ void sortposition(float* positions, float* fitness, const int func_num) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < popsize * dimension) {
        //对齐positions
        float fit = benchmark_func(&positions[blockIdx.x * dimension], func_num);
        float position = positions[idx];
        for (int i = 0; i < popsize; i++) {
            if (fitness[i] == fit) {
                positions[i * dimension + threadIdx.x] = position;
            }
        }
    }
}
__global__ void learning(float* positions, float* fitness,
    curandState* states, const int func_num) {

    //寄存器255
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int s_randid1;
    __shared__ int s_randid2;
    __shared__ int s_randid3;
    __shared__ int s_randid4;
    __shared__ float s_visual1[dimension];
    __shared__ float SF;
    __shared__ float rate1;
    __shared__ float rate2;
    __shared__ float rate3;
    __shared__ float rate4;
    __shared__ float s_newpositoin[dimension];
    if (idx < popsize * dimension) {
        float fit = fitness[blockIdx.x];
        float position = positions[idx];

        if (threadIdx.x == 0) {
            s_randid1 = (int)getrand(threadIdx.x, 1, 5, states);
            s_randid2 = (int)getrand(threadIdx.x, popsize - Bnum, popsize, states);
            while ((s_randid3 = (int)getrand(threadIdx.x, Enum, popsize - Bnum, states)) && (s_randid4 = (int)getrand(threadIdx.x, Enum, popsize - Bnum, states))) {
                if (s_randid3 != s_randid4)
                    break;
            }
        }
        __syncthreads();

        s_visual1[threadIdx.x] = positions[threadIdx.x] - positions[s_randid1 * dimension + threadIdx.x];
        __syncthreads();
        rate1 = go_norm(s_visual1);
        float newx = s_visual1[threadIdx.x] * rate1;

        s_visual1[threadIdx.x] = positions[threadIdx.x] - positions[s_randid2 * dimension + threadIdx.x];
        __syncthreads();
        rate2 += go_norm(s_visual1);
        newx += s_visual1[threadIdx.x] * rate2;

        s_visual1[threadIdx.x] = positions[s_randid1 * dimension + threadIdx.x] - positions[s_randid2 * dimension + threadIdx.x];
        __syncthreads();
        rate3 += go_norm(s_visual1);
        newx += s_visual1[threadIdx.x] * rate3;

        s_visual1[threadIdx.x] = positions[s_randid3 * dimension + threadIdx.x] - positions[s_randid4 * dimension + threadIdx.x];
        __syncthreads();
        rate4 += go_norm(s_visual1);
        newx += s_visual1[threadIdx.x] * rate4;

        SF = powf(fit / fitness[popsize - 1], 1.25);
        newx = position + SF * newx / (rate1 + rate2 + rate3 + rate4);

        if (newx > xmax) newx = xmax;
        else if (newx < xmin) newx = xmin;
        s_newpositoin[threadIdx.x] = newx;
        if (threadIdx.x == 0)
            s_randid1 = getrand(threadIdx.x, 0, 1, states);
        __syncthreads();
        float newfit = benchmark_func(s_newpositoin, func_num);
        if (newfit < fit || s_randid1 < 0.001 && blockIdx.x != 0) {
            fit = newfit;
            position = newx;
        }
        if (threadIdx.x == 0)
            fitness[blockIdx.x] = fit;
        positions[idx] = position;
    }
}

__global__ void gradientGuid(float* positions, float* fitness, const int func_num) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float gradx[dimension];
    __shared__ float newgradx[dimension];
    if (idx < Enum * dimension) {
        float gradfit = fitness[blockIdx.x];
        gradx[threadIdx.x] = positions[idx];
        float gradientx[dimension];
        float gradw = powf(gradfit / fitness[popsize - 1], 1.25);
        __syncthreads;
        for (int i = 0; i < dimension; i++)
        {
            gradientx[i] = gradx[i];
        }
        gradientx[threadIdx.x] += epsilon;
        float gradient1 = benchmark_func(gradientx, func_num);
        gradientx[threadIdx.x] -= 2 * epsilon;
        float gradient = (gradient1 - benchmark_func(gradientx, func_num)) / 2 / epsilon;
        for (int iter = 0; iter < 30; iter++)
        {
            gradw *= 0.8;
            newgradx[threadIdx.x] = gradx[threadIdx.x] - gradient * gradw;
            newgradx[threadIdx.x] = (newgradx[threadIdx.x] < xmin) ? xmin : newgradx[threadIdx.x];
            newgradx[threadIdx.x] = (newgradx[threadIdx.x] > xmax) ? xmax : newgradx[threadIdx.x];
            __syncthreads;
            float newgradfit = benchmark_func(newgradx, func_num);
            if (gradfit > newgradfit) {
                gradfit = newgradfit;
                gradx[threadIdx.x] = newgradx[threadIdx.x];
            }
        }
        if (gradfit != fitness[blockIdx.x]) {
            if (threadIdx.x == 0)
                fitness[blockIdx.x] = gradfit;
            positions[idx] = gradx[threadIdx.x];
        }
    }
}

__global__ void reflection(float* positions, float* fitness,
    const int Fes,
    curandState* states, const int func_num) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float s_newx[dimension];
    __shared__ float s_rand;
    if (idx < popsize * dimension) {
        float x = positions[idx];
        float fit = fitness[blockIdx.x];
        float newx = x;
        if (getrand(threadIdx.x, 0, 1, states) < P3) {
            float R = positions[(int)getrand(threadIdx.x, 0, 5, states) * dimension + threadIdx.x];
            newx = x + (R - x) * getrand(threadIdx.x, 0, 1, states);
            if (newx < xmin)newx = xmin;
            else if (newx > xmax)newx = xmax;
            if (getrand(threadIdx.x, 0, 1, states) < 0.01 + (0.1 - 0.001) * (1 - Fes / MaxFEs))
                newx = getrand(threadIdx.x, xmin, xmax, states);
        }
        s_newx[threadIdx.x] = newx;
        s_rand = getrand(threadIdx.x, 0, 1, states);
        __syncthreads();
        float newfit = benchmark_func(s_newx, func_num);
        if (newfit < fit || s_rand < P2 && blockIdx.x != 0) {
            fit = newfit;
            x = newx;
        }
        positions[idx] = x;
        if (threadIdx.x == 0)
            fitness[blockIdx.x] = fit;
    }
}

int cudaGGOnoneStruct(const int func_num) {
    // 分配设备内存
    float* d_positions, * d_fitness;
    float* d_temp;
    curandState* d_states_particle;
    cudaMalloc(&d_states_particle, dimension * sizeof(curandState));
    cudaMalloc(&d_positions, popsize * dimension * sizeof(float));
    cudaMalloc(&d_fitness, popsize * sizeof(float));
    cudaMalloc(&d_temp, popsize * sizeof(float));

    // 配置CUDA网格和线程块
    dim3 block(BLOCK_SIZE);
    dim3 grid((popsize * dimension + BLOCK_SIZE - 1) / BLOCK_SIZE);

    init_particle << <grid, block >> > (d_positions, d_fitness, d_states_particle, time(NULL), func_num);
    cudaDeviceSynchronize();
    for (int Fes = 0; Fes < MaxFEs; ++Fes) {
        int width = 1;
        while (width < popsize) {
            mergeSortKernel << <(popsize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_fitness, d_temp, width, popsize);
            cudaDeviceSynchronize();
            width *= 2;
        }
        sortposition << <grid, block >> > (d_positions, d_fitness, func_num);
        cudaDeviceSynchronize();
        learning << <grid, block >> > (d_positions, d_fitness, d_states_particle, func_num);
        cudaDeviceSynchronize();
        gradientGuid << <(Enum * dimension + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_positions, d_fitness, func_num);
        cudaDeviceSynchronize();
        reflection << <grid, block >> > (d_positions, d_fitness, Fes, d_states_particle, func_num);
        cudaDeviceSynchronize();
    }
    // 将结果复制回主机
    float h_positions[dimension];
    cudaMemcpy(h_positions, d_positions, dimension * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << std::left << std::setw(12) << cpu_benchmark_func(h_positions, func_num);

    // 清理内存
    cudaFree(d_positions);
    cudaFree(d_fitness);
    cudaFree(d_temp);
    cudaFree(d_states_particle);
    return 0;
}

