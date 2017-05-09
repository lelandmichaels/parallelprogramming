double f(double  x){
  return x*x;
}

main(){
  int a = 1;
  int b = 2;
  int n = 1000000;
  double h = (b-a)/(double)n;
  double sum = 0.0;
  sum += (f(a) + f(b))/2.0;
  for(int i = 1; i<n; i++){
    double  x_i = a+i*h;
    sum += f(x_i); 
  }
  sum = h*sum;
  printf("%f\n", sum);

}

