#include "ggo.h"

//cpu版本

// F1: Sphere's Problem
float serial_f1(const float* x) {
    float sum = 0;
    for (int i = 0; i < dimension; i++)
        sum += (x[i] - i) * (x[i] - i);
    return sum;
}

// F2: Schwefel's Problem 2.21
float serial_f2(float* x) {
    float max_val = fabsf(x[0]);
    for (int i = 1; i < dimension; ++i) {
        if (fabsf(x[i]) > max_val) {
            max_val = fabsf(x[i]);
        }
    }
    return max_val;
}

// F3: Step Function
float serial_f3(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += (floor(x[i] + 0.5)) * (floor(x[i] + 0.5));
    }
    return sum;
}

// F4: Ackley's Function
float serial_f4(float* x) {
    float sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum1 += x[i] * x[i];
        sum2 += cosf(2 * PI * x[i]);
    }
    float a = 20, b = 0.2, c = 2 * PI;
    return -a * expf(-b * sqrtf(sum1 / dimension)) - expf(sum2 / dimension) + a + expf(1);
}

// F5: Rastrigin's Function
float serial_f5(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i] - 10 * cosf(2 * PI * x[i]);
    }
    return 10 * dimension + sum;
}

// F6: Griewank's Function
float serial_f6(float* x) {
    float sum = 0.0;
    float prod = 1.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i];
        prod *= cosf(x[i] / sqrtf(i + 1));
    }
    return sum / 4000 - prod + 1;
}

// F7: Sphere Function
float serial_f7(float* x) {
    float sum = 0.0;
    for (int i = 0; i < dimension; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

// 测试函数接口
float serial_benchmark_func(float* x, int func_num) {
    switch (func_num) {
    case 1: return serial_f1(x);
    case 2: return serial_f2(x);
    case 3: return serial_f3(x);
    case 4: return serial_f4(x);
    case 5: return serial_f5(x);
    case 6: return serial_f6(x);
    case 7: return serial_f7(x);

    default: return -1.0; // Invalid function number
    }
}


// 交换两个元素的值
void serial_swap(float* a, float* b, float positions_a[], float positions_b[]) {
    float t = *a;
    *a = *b;
    *b = t;
    for (int i = 0; i < dimension; i++)
    {
        t = positions_a[i];
        positions_a[i] = positions_b[i];
        positions_b[i] = t;
    }
}
// 分区函数，选择一个元素作为基准，并将小于基准的元素放在左边，大于基准的元素放在右边
int serial_partition(float arr[], int low, int high, float positions[][dimension]) {
    float pivot = arr[high]; // 选择最后一个元素作为基准
    int i = (low - 1); // i指向比基准小的元素的最后一个位置

    for (int j = low; j <= high - 1; j++) {
        // 如果当前元素小于或等于基准
        if (arr[j] <= pivot) {
            i++; // 增加小于基准的元素的计数
            serial_swap(&arr[i], &arr[j], positions[i], positions[j]); // 交换arr[i]和arr[j]
        }
    }
    serial_swap(&arr[i + 1], &arr[high], positions[i + 1], positions[high]); // 将基准元素放到正确的位置
    return (i + 1); // 返回基准元素的最终位置
}
// 快速排序函数
void serial_quickSort(float arr[], int low, int high, float positions[][dimension]) {
    if (low < high) {
        // pi是分区索引，arr[pi]现在在正确的位置
        int pi = serial_partition(arr, low, high, positions);

        // 分别递归地对基准左右两边的子数组进行快速排序
        serial_quickSort(arr, low, pi - 1, positions);
        serial_quickSort(arr, pi + 1, high, positions);
    }
}
void computeGradient(float* x, float* arr, const int func_num) {
    float ref = 0;
    float newx[dimension];
    for (int i = 0; i < dimension; i++) {
        newx[i] = x[i];
    }
    for (int i = 0; i < dimension; i++)
    {
        newx[i] += epsilon;
        float a = serial_benchmark_func(newx, func_num);
        newx[i] -= 2 * epsilon;
        float b = serial_benchmark_func(newx, func_num);
        arr[i] = (a - b) / epsilon / 2;
        newx[i] += epsilon;
    }
}

// 返回指定区间的随机整数
int serial_getRandom_int(float low, float high)
{
    return ((high - low + 1) * (int)rand() / RAND_MAX) + low;
}
//随机Worst_X、Better_X和两个随机粒子L1、L2
void serial_d_Gap(int i, int a[])
{

    a[0] = serial_getRandom_int(popsize - P1, popsize - 1);
    a[1] = serial_getRandom_int(1, P1 - 1);
    a[2] = serial_getRandom_int(1, popsize - 1);
    a[3] = serial_getRandom_int(1, popsize - 1);
    while (a[2] == i || a[3] == i || a[2] == a[3])
    {
        if (a[2] == i)
            a[2] = serial_getRandom_int(1, popsize - 1);
        else if (a[2] == a[3])
            a[3] = serial_getRandom_int(1, popsize - 1);
        else
            a[3] = serial_getRandom_int(1, popsize - 1);
    }
}
//norm()函数的实现
float serial_norm(float* a)
{
    float c = 0;
    for (int i = 0; i < dimension; ++i)
        c += a[i] * a[i];
    return sqrt(c);
}
//寻找数组中的最值,model:min:0，max:1
float serial_best_value(float* arr, const int n, const int model)
{
    float res = 0;
    switch (model) {
    case 0://求最小值
        res = INFINITY;
        for (int i = 0; i < n; i++) {
            if (res > arr[i]) res = arr[i];
        }break;
    case 1:
        res = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (res < arr[i]) res = arr[i];
        }break;
    }
    return res;
}
//更新粒子位置并保证在边界内
void serial_update_positions(float* position1, float* position2, float* position3
    , float* position4, float* position5, float* position6)
{
    for (int i = 0; i < dimension; i++) {
        position6[i] = position1[i] + position2[i] + position3[i] + position4[i] + position5[i];
    }
    for (int i = 0; i < dimension; i++) {
        if (position6[i] > xmax) position6[i] = xmax;
        else if (position6[i] < xmin) position6[i] = xmin;
    }
}
int cpuGGO(const int func_num)
{
    float positions[popsize][dimension];
    float fitness[popsize];
    int FEs = 0;
    float gbestX[dimension];
    float gbestfitness = INFINITY;
    float gbesthistory[MaxFEs];
    int a[4];//该四个粒子分别代表Worst_X、Better_X、随机粒子L1、随机粒子L2
    float temp_positions[dimension];
    srand((unsigned int)time(NULL));
    for (int i = 0; i < popsize * dimension; i++)//初始化位置
    {
        positions[i / dimension][i % dimension] = ((float)rand() / RAND_MAX) * (xmax - xmin) + xmin;
    }
    for (int i = 0; i < popsize; i++)//适应度及最佳适应度计算
    {
        for (int j = 0; j < dimension; j++)
            temp_positions[j] = positions[i][j];
        fitness[i] = serial_benchmark_func(temp_positions, func_num);
        if (gbestfitness > fitness[i])
        {
            gbestfitness = fitness[i];
            for (int j = 0; j < dimension; j++)
                gbestX[j] = positions[i][j];
        }
        //printf("\nFEs: %d，fitness error: %f\npartcile：", FEs, gbestfitness);
        //for(int kk=0;kk<dimension;kk++)
            //printf("%8.2f ", gbestX[kk]);
    }
    gbesthistory[FEs++] = gbestfitness;
    while (1)
    {
        float Best_X[dimension];
        float Worst_X[dimension];
        float Better_X[dimension];
        float L1[dimension];
        float L2[dimension];
        int a[4];
        serial_quickSort(fitness, 0, popsize - 1, positions);
        for (int j = 0; j < dimension; j++)
            Best_X[j] = positions[0][j];
        //Learning phase
        for (int i = 0; i < popsize; i++)
        {
            serial_d_Gap(i, a);
            for (int j = 0; j < dimension; j++)
            {
                Worst_X[j] = positions[a[0]][j];
                Better_X[j] = positions[a[1]][j];
                L1[j] = positions[a[2]][j];
                L2[j] = positions[a[3]][j];
            }
            float Gap1[dimension];
            float Gap2[dimension];
            float Gap3[dimension];
            float Gap4[dimension];
            for (int j = 0; j < dimension; j++)
            {
                Gap1[j] = Best_X[j] - Better_X[j];
                Gap2[j] = Best_X[j] - Worst_X[j];
                Gap3[j] = Better_X[j] - Worst_X[j];
                Gap4[j] = L1[j] - L2[j];
            }
            float Distance1 = serial_norm(Gap1);
            float Distance2 = serial_norm(Gap2);
            float Distance3 = serial_norm(Gap3);
            float Distance4 = serial_norm(Gap4);
            float SumDistance = Distance1 + Distance2 + Distance3 + Distance4;
            float LF1 = Distance1 / SumDistance;
            float LF2 = Distance2 / SumDistance;
            float LF3 = Distance3 / SumDistance;
            float LF4 = Distance4 / SumDistance;
            float SF = (fitness[i] / serial_best_value(fitness, popsize, 1));
            float KA1[dimension];
            float KA2[dimension];
            float KA3[dimension];
            float KA4[dimension];
            for (int j = 0; j < dimension; j++)
            {
                KA1[j] = LF1 * SF * Gap1[j];
                KA2[j] = LF2 * SF * Gap2[j];
                KA3[j] = LF3 * SF * Gap3[j];
                KA4[j] = LF4 * SF * Gap4[j];
            }
            float newx[dimension];
            for (int j = 0; j < dimension; j++)
                temp_positions[j] = positions[i][j];
            //Clipping
            serial_update_positions(temp_positions, KA1, KA2, KA3, KA4, newx);
            float newfitness = serial_benchmark_func(newx, func_num);

            //梯度
            if (i < Enum) {
                float gradient[dimension];
                computeGradient(positions[i], gradient, func_num);
                float gradx[dimension];
                for (int j = 0; j < dimension; j++)
                {
                    gradx[j] = positions[i][j];
                }
                float gradfit = serial_benchmark_func(gradx, func_num);
                int T = 30;
                float gradw = SF;
                float newgradx[dimension];
                for (int iter = 0; iter < T; iter++)
                {
                    gradw *= 0.8;
                    for (int j = 0; j < dimension; j++)
                    {
                        newgradx[j] = gradx[j] - gradw * gradient[j];
                        newgradx[j] = max(newgradx[j], xmin);
                        newgradx[j] = min(newgradx[j], xmax);
                    }
                    float newgradfit = serial_benchmark_func(newgradx, func_num);
                    if (newgradfit < gradfit) {
                        gradfit = newgradfit;
                        for (int j = 0; j < dimension; j++)
                        {
                            gradx[j] = newgradx[j];
                        }
                    }
                }
                if (gradfit < newfitness) {
                    newfitness = gradfit;
                    for (int j = 0; j < dimension; j++)
                    {
                        newx[j] = newgradx[j];
                    }
                }
            }
            //Update
            if (fitness[i] > newfitness)
            {
                fitness[i] = newfitness;
                for (int j = 0; j < dimension; j++)
                    positions[i][j] = newx[j];
            }
            else if (((float)rand() / RAND_MAX) < P2 && i != 0)
            {
                fitness[i] = newfitness;
                for (int j = 0; j < dimension; j++)
                    positions[i][j] = newx[j];
            }
            if (gbestfitness > fitness[i])
            {
                gbestfitness = fitness[i];
                for (int j = 0; j < dimension; j++)
                    gbestX[j] = positions[i][j];
            }
            //if (FEs >= MaxFEs) break;
            //gbesthistory[FEs++] = gbestfitness;

            //printf("\nFEs: %d，fitness error: %f\npartcile：", FEs, gbestfitness);
            //for (int kk = 0; kk < dimension; kk++)
                //printf("%8.2f ", gbestX[kk]);
        }
        if (FEs >= MaxFEs) break;

        //Reflection phase
        for (int i = 0; i < popsize; i++)
        {
            float AF;
            float newx[dimension];
            for (int j = 0; j < dimension; j++)
                newx[j] = positions[i][j];
            for (int j = 0; j < dimension; j++)
                if (((float)rand() / RAND_MAX) < P3)
                {
                    newx[j] = positions[i][j] + (positions[serial_getRandom_int(0, P1 - 1)][j] - positions[i][j]) * ((float)rand() / RAND_MAX);
                    AF = (0.01 + (0.1 - 0.01) * (1 - FEs / MaxFEs));
                    if (((float)rand() / RAND_MAX) < AF)
                        newx[j] = xmin + (xmax - xmin) * ((float)rand() / RAND_MAX);
                }
            //Clipping
            for (int j = 0; j < dimension; j++)
                if (newx[j] > xmax)
                    newx[j] = xmax;
                else if (newx[j] < xmin)
                    newx[j] = xmin;
            float newfitness = serial_benchmark_func(newx, func_num);
            //Update
            if (fitness[i] > newfitness)
            {
                fitness[i] = newfitness;
                for (int j = 0; j < dimension; j++)
                    positions[i][j] = newx[j];
            }
            else if ((float)rand() / RAND_MAX < P2 && i != 1)
            {
                fitness[i] = newfitness;
                for (int j = 0; j < dimension; j++)
                    positions[i][j] = newx[j];
            }
            if (gbestfitness > fitness[i])
            {
                gbestfitness = fitness[i];
                for (int j = 0; j < dimension; j++)
                    gbestX[j] = positions[i][j];
            }
            if (FEs >= MaxFEs) break;
            //gbesthistory[FEs++] = gbestfitness;
            //printf("\nFEs: %d，fitness error: %f\npartcile：", FEs, gbestfitness);
            //for (int kk = 0; kk < dimension; kk++)
                //printf("%8.2f ", gbestX[kk]);
        }
        if (FEs >= MaxFEs) break;
        gbesthistory[FEs++] = gbestfitness;
    }
    std::cout << std::left << std::setw(12) << gbesthistory[MaxFEs - 1];

    //std::cout.unsetf(std::ios_base::floatfield); // 取消之前的浮点数格式设置
    //std::cout << "best position : " << std::endl;
    //for (int i = 0; i < dimension; ++i) {
    //    std::cout << std::left << std::setw(12) << gbestX[i];
    //}
    return 0;
}
