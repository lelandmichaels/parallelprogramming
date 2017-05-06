#include <iostream>
#include <math.h>
// Kernel function to calculate trapizoidal sum
__global__
void trap(int a, float h, double* sum){
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
  
  trap<<<1, 1>>>(N, x, y);

  cudaDeviceSynchronize();
  
  *sum = h*sum;
  printf("%f\n", sum);
  return 0;
}
