#pragma once

#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <iomanip>//����������ͷ�ļ���ʹ�ø�ʽ��������
#include <math.h>

inline float min(float a, float b);
inline float max(float a, float b);

const int test_iter = 10;
#define PI 3.14159265358979323846
#define epsilon 1e-20//������

const int popsize = 512;
const int dimension = 32;
const int xmax = dimension;
const int xmin = 0;
const int MaxFEs = 5000;
const int BLOCK_SIZE = dimension;

const int P1 = 5;
#define P2 0.001
#define P3 0.3
const int Enum = 10;//��Ӣ����
const int Bnum = 4;//������


int cpuGGO(const int func_num);
int cudaGGOnoneStruct(const int func_num);
