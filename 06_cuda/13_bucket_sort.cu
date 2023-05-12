#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void create_bucket(int *bucket,int *key,int n,int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<range)
    bucket[i]=0;
  if(i>=n) return;
  atomicAdd(&(bucket[key[i]]),1);
}

__global__ void create_offset(int *offset,int *bucket,int *temp,int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  grid_group grid = this_grid();
  if(i<range)
    offset[i]=0;
  if(1<= i < range)
    offset[i]=bucket[i-1];
  grid.sync();
  for(int j=1; j<range; j<<=1) {
    if(0<=i && i<range)
      temp[i] = offset[i];
    grid.sync();
    if(j<=i && i<range)
        offset[i] += temp[i-j];
    grid.sync();
  }
}

__global__ void create_result(int *offset,int *bucket, int *key, int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<range){
    int j = offset[i];
    for (int k=0; k<bucket[i]; k++) {
      key[j++] = i; // Atomicであることが保証されている．
    }
    bucket[i]=0;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  cudaMallocManaged(&key, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  const int M = 1024;
  const int GRID = 2;

  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  int *offset;
  cudaMallocManaged(&offset, range*sizeof(int));
  int *temp;
  cudaMallocManaged(&temp, range*sizeof(int));
  cudaDeviceSynchronize();

  void *args0[] = {(void *)&bucket,  (void *)&key, (void *)&n,(void *)&range};
  cudaLaunchCooperativeKernel((void*)create_bucket, GRID, M/GRID, args0);
  cudaDeviceSynchronize();
  
  void *args1[] = {(void *)&offset,  (void *)&bucket, (void *)&temp,(void *)&range};
  cudaLaunchCooperativeKernel((void*)create_offset, GRID, M/GRID, args1);
  cudaDeviceSynchronize();
  cudaFree(temp);

  void *args2[] = {(void *)&offset,  (void *)&bucket, (void *)&key,(void *)&range};
  cudaLaunchCooperativeKernel((void*)create_result, GRID, M/GRID, args2);
  cudaDeviceSynchronize();
  cudaFree(bucket);
  cudaFree(offset);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  cudaFree(key);
  printf("\n");
}