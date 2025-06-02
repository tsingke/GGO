#include "ggo.h"

#include <time.h>
#include <cuda_runtime.h>

inline float min(float a, float b) {
	return (a < b) ? a : b;
}
inline float max(float a, float b) {
	return (a > b) ? a : b;
}

int main() {
	for (int f = 1; f <= 7; f++)
	{

		printf("CEC2017 F%d:\n", f);
	
		cudaEvent_t start, stop;
		float elapsed_time1, mean1 = 0, min1 = INFINITY, max1 = 0,
			elapsed_time2, mean2 = 0, min2 = INFINITY, max2 = 0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		for (int i = 0; i < test_iter; i++)
		{
			cudaEventRecord(start);
			cudaEventQuery(start);

			cudaGGOnoneStruct(f);

			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed_time1, start, stop);
			min1 = min(min1, elapsed_time1);
			max1 = max(max1, elapsed_time1);
			mean1 += elapsed_time1;
			printf("\n");
		}

		printf("\nd_Mean_Time = %g ms.\n", mean1 /= test_iter);
		printf("\nd_Min_Time = %g ms.\n", min1);
		printf("\nd_Max_Time = %g ms.\n", max1);


		for (int i = 0; i < test_iter; i++)
		{
			cudaEventRecord(start);
			cudaEventQuery(start);

			cpuGGO(f);

			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed_time2, start, stop);
			min2 = min(min2, elapsed_time2);
			max2 = max(max2, elapsed_time2);
			mean2 += elapsed_time2;
			printf("\n");
		}

		printf("\nh_Mean_Time = %g ms.\n", mean2 /= test_iter);
		printf("\nh_Min_Time = %g ms.\n", min2);
		printf("\nh_Max_Time = %g ms.\n", max2);


		printf("mean speedup = %g .\n \n", mean2 / mean1);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	
	return 0;
}
