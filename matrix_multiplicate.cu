#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


__host__ void MatrixInit(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            int random_value = rand();                
            float normalized = (float)random_value / RAND_MAX;  
            M[i * p + j] = (normalized * 2.0f) - 1.0f;
        }
    }
}

__host__ void MatrixPrint(float *M, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%f ", M[i * p + j]);
        }
        printf("\n");
    }
}

__host__ void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2 [i * p + j];
        }
    }

}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < p) {
        int index = idx * p + idy;
        Mout[index] = M1[index] + M2[index];
    }
}    

__host__ void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0; 
            for (int k = 0; k < n; k++){
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        Mout[idx * n + idy] = 0; 
        for (int k = 0; k < n; k++){
            Mout[idx * n + idy] += M1[idx * n + k] * M2[k * n + idy];
        }    
    }

}


int main(int argc, char *argv[]){

    assert(argc == 3); 
    int n = atoi(argv[1]);
    int p = atoi(argv[2]);

    
    float *M1;
    float *M2;
    float *Madd;
    float *Mmult;
    float *d_M1;
    float *d_M2;
    float *d_Madd;
    float *d_Mmult;


    M1 = (float*)malloc(sizeof(float) * n * p);
    M2 = (float*)malloc(sizeof(float) * n * p);
    Madd = (float*)malloc(sizeof(float) * n * p);
    Mmult = (float*)malloc(sizeof(float) * n * n);



    printf("M1\n");
    time_t a = clock();
    MatrixInit(M1, n, p);
    printf("Exécution de Init, %f sec", (double)(clock()-a) / CLOCKS_PER_SEC);
    printf("\n");
    a = clock();
    //MatrixPrint(M1, n, p);
    printf("Exécution de Print, %f sec", (double)(clock()-a) / CLOCKS_PER_SEC);
    //printf("\nM2\n");
    MatrixInit(M2, n, p);
    //MatrixPrint(M2, n, p);
    printf("\nMadd CPU\n");
    a = clock();
    MatrixAdd(M1, M2, Madd, n, p);
    printf("Exécution de Add, %f sec", (double)(clock()-a) / CLOCKS_PER_SEC);
    //MatrixPrint(Madd, n, p);
    printf("\nMmult CPU\n");
    a = clock();
    MatrixMult(M1, M2, Mmult, n);
    printf("Exécution de Mult, %f sec \n", (double)(clock()-a) / CLOCKS_PER_SEC);
    //MatrixPrint(Mmult, n, n);


    cudaMalloc((void **)&d_M1, sizeof(float) * n * p);
    cudaMalloc((void **)&d_M2, sizeof(float) * n * p);
    cudaMalloc((void **)&d_Madd, sizeof(float) * n * p);
    cudaMalloc((void **)&d_Mmult, sizeof(float) * n * n);
    cudaMemcpy(d_M1, M1, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    dim3 blockDim(n);
    dim3 gridDim(1, p);
    cudaMatrixAdd<<<gridDim, blockDim>>>(d_M1, d_M2, d_Madd, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(Madd, d_Madd, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    printf("\nMadd GPU\n");
    //MatrixPrint(Madd, n, p);

    dim3 blockDimM(n);
    dim3 gridDimM(1, n);
    cudaMatrixMult<<<gridDimM, blockDimM>>>(d_M1, d_M2, d_Mmult, n);
    cudaDeviceSynchronize();

    cudaMemcpy(Mmult, d_Mmult, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    printf("\nMmult GPU \n");
    //MatrixPrint(Mmult, n, n);


    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Madd);
    cudaFree(d_Mmult);
    

    free(M1);
    free(M2);
    free(Madd);
    free(Mmult);


    return (0);

}