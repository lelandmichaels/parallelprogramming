#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void writeheader(int N, int end) {
	FILE *fp;
	fp = fopen("outfile.pgm", "w");
	if (fp == NULL) {
		printf("sorry can't open outfile.pgm. Terminating.\n");
		exit(1);
	}
	else {
		// print a table header
		fprintf(fp, "%s\n%d %d\n%s\n", "P2", N, end, "255");
		fclose(fp);
	}
}

void writerow(int N, double rawdata[]) {
	FILE *fp;
	fp = fopen("outfile.pgm", "a");
	if (fp == NULL) {
		printf("sorry can't open outfile.pgm. Terminating.\n");
		exit(1);
	}
	else {
		for (int i=0; i<N; i++) {
			int val = rawdata[i]*127+127;
			fprintf(fp,"%d ",val);
		}
		fprintf(fp,"\n");
		fclose(fp);
	}
}

int main(int argc, char *argv[]) {
	int comm_sz;
	int my_rank;
	int N = 6;
	int localN;
	int end = 120;//end=20N is roughly 1 period
	int writeoutput = 1;//0 for false
	
	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	
	localN = N/comm_sz;//assuming divisible

	if (N%comm_sz!=0) {
		printf("ERROR N must be divisible by comm_sz. Terminating.");
		exit(1);
	}
	if (localN<2) {
		printf("ERROR N (%d) must be at least twice as large as number of processes (%d). Terminating.",N,comm_sz);
		exit(1);
	}
	
	double * f0 = (double *)malloc(localN*sizeof(double));
	double * f1 = (double *)malloc(localN*sizeof(double));
	double * f2 = (double *)malloc(localN*sizeof(double));
	double * temp;// = (double *)malloc(localN*sizeof(double));
	
	double forOutput[N];
	
	//keep original pointers so we can free later
	double * originalf0 = f0;
	double * originalf1 = f1;
	double * originalf2 = f2;
	
	double localx0 = 1.0/(N-1)*my_rank*localN;
	
	for (int i=0; i<localN; i++) {
		double x = localx0 + (double)i*1.0/(N-1);
		f0[i] = sin(M_PI*x);
		f1[i] = sin(M_PI*x);
	}
	if (my_rank==0) {
		f0[0] = 0;
		f1[0] = 0;
		f2[0] = 0;
	}
	if (my_rank==comm_sz-1) {
		f0[localN-1] = 0;
		f1[localN-1] = 0;
		f2[localN-1] = 0;
	}
	
	//write header and f1
	if (writeoutput) {
		MPI_Gather(f1,localN,MPI_DOUBLE,forOutput,localN,MPI_DOUBLE,0,MPI_COMM_WORLD);
		if (my_rank==0)
		{
			writeheader(N,end);
			writerow(N,forOutput);
		}
	}
	
	double leftneighbor=0;
	double rightneighbor=0;
	
	int partnerRight = my_rank+1;
	int partnerLeft = my_rank-1;
	if (partnerRight==comm_sz) {
		partnerRight=MPI_PROC_NULL;
	}
	if (partnerLeft==-1) {
		partnerLeft=MPI_PROC_NULL;
	}

	//main loop
	int step = 2;//current step
	while (step<=end) {
		//send right
		MPI_Sendrecv(&f1[localN-1],1,MPI_DOUBLE,partnerRight,0,&leftneighbor,1,MPI_DOUBLE,partnerLeft,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		//send left
		MPI_Sendrecv(&f1[0],1,MPI_DOUBLE,partnerLeft,0,&rightneighbor,1,MPI_DOUBLE,partnerRight,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		//put next step in f2:
		//compute interior of my domain
		for (int i=1; i<localN-1; i++) {
			f2[i] = 0.01*(f1[i-1]-2*f1[i]+f1[i+1]) + 2*f1[i] - f0[i];
		}
		
		//compute left edge of my domain
		if (my_rank!=0) {
			int i=0;
			f2[i] = 0.01*(leftneighbor-2*f1[i]+f1[i+1]) + 2*f1[i] - f0[i];
		}
		//compute right edge of my domain
		if (my_rank!=comm_sz-1) {
			int i=localN-1;
			f2[i] = 0.01*(f1[i-1]-2*f1[i]+rightneighbor) + 2*f1[i] - f0[i];
		}
		
		//write output
		if (writeoutput) {
			MPI_Gather(f2,localN,MPI_DOUBLE,forOutput,localN,MPI_DOUBLE,0,MPI_COMM_WORLD);
			if (my_rank==0) {
				writerow(N,forOutput);
			}
		}

		//rearrange pointers for next step
		temp=f0;
		f0=f1;
		f1=f2;
		f2=temp;

		step++;
	}
	
	//free memory
	//can't free unless reassign to original, ok in this program, but possible memory leak in others
//	free(f0);
//	free(f1);
//	free(f2);
	f0 = originalf0;
	f1 = originalf1;
	f2 = originalf2;
	free(f0);
	free(f1);
	free(f2);

	MPI_Finalize();

	return 0;
}
