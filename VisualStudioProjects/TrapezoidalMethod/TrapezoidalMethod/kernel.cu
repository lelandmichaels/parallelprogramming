#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>
// Kernel function to calculate trapizoidal sum
double f(double x){
  return x*x;
}

__global__
void trap(int a, int n, float h, double* sum){
  for(int i = 1; i<n; i++){
    double x_i = (double)a+i*h;
    *sum += f(x_i); 
  }
}

int main(void){

  int a = 1;
  int b = 2;
  int n = 100;
  double  h = (b-a)/(double)n;
  double *sum = 0.0;
  *sum += (f(a) + f(b))/2.0;
  
  trap<<<1, 1>>>(a, n, h, *sum);

  cudaDeviceSynchronize();
  
  *sum = h*sum;
  printf("%f\n", sum);
  return 0;
}
