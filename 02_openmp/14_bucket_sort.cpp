#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
#pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++){
#pragma omp atomic update
    bucket[key[i]]++;
  }
  std::vector<int> offset(range,0);
//   for (int i=1; i<range; i++) 
    // offset[i] = offset[i-1] + bucket[i-1];
/*   
/  offset[i]= offset[i-1] + bucket[i-1] (1<=i<range)
/  offset[0]=0
/  then offset[n]-offset[0] = \sum_{i=1}^{n} bucket[i-1] (1<=i<range)
/  then offset[n] = \sum_{i=0}^{n-1} bucket[i](1<=i<range)
*/
  std::vector<int> temp(range,0);
#pragma omp parallel for
  for(int i=1; i<range; i++){
    offset[i] = bucket[i-1];
  }
#pragma omp parallel
  for(int j=1; j<range; j<<=1) {
#pragma omp for
    for(int i=0; i<range; i++)
      temp[i] = offset[i];
#pragma omp for
    for(int i=j; i<range; i++)
      offset[i] += temp[i-j];
  }

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
#pragma omp parallel for
    for (int k=0; k<bucket[i]; k++) {
      key[j++] = i;
    }
    bucket[i]=0;
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
