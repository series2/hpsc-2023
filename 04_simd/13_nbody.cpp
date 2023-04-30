#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#define SIMD true
int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  #if SIMD
  printf("with simd\n");
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  for(int i=0; i<N; i++) {
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);

    __m256 rxvec = _mm256_sub_ps(xivec,xvec);
    __m256 rx2vec = _mm256_mul_ps(rxvec,rxvec);

    __m256 ryvec = _mm256_sub_ps(yivec,yvec);
    __m256 ry2vec = _mm256_mul_ps(ryvec,ryvec);

    __m256 r2vec = _mm256_add_ps(rx2vec,ry2vec);
    __m256 rsqrt = _mm256_rsqrt_ps(r2vec);

    __m256 common_denom = _mm256_mul_ps(mvec,_mm256_mul_ps(rsqrt,_mm256_mul_ps(rsqrt,rsqrt)));

    // どうやらrsqrtは0除算で実行時エラーを出さないようなので，そこまでは計算する．
    // r2vecが小さすぎる要素を自分自身への力と見做してmaskする．
    __m256 epsilon = _mm256_set1_ps(1e-15);
    __m256 mask = _mm256_cmp_ps(r2vec,epsilon,_CMP_GT_OQ);
    common_denom = _mm256_blendv_ps(_mm256_setzero_ps(),common_denom,mask);

    __m256 sub_force_x=_mm256_mul_ps(rxvec,common_denom);
    __m256 sub_force_y=_mm256_mul_ps(ryvec,common_denom);
    float reduction_sub_force_x[N],reduction_sub_force_y[N];

    // reduction
    __m256 temp_vec = _mm256_permute2f128_ps(sub_force_x,sub_force_x,1);
    temp_vec = _mm256_add_ps(temp_vec,sub_force_x);
    temp_vec = _mm256_hadd_ps(temp_vec,temp_vec);
    temp_vec = _mm256_hadd_ps(temp_vec,temp_vec);
    _mm256_store_ps(reduction_sub_force_x, temp_vec);

    // reduction
    temp_vec = _mm256_permute2f128_ps(sub_force_y,sub_force_y,1);
    temp_vec = _mm256_add_ps(temp_vec,sub_force_y);
    temp_vec = _mm256_hadd_ps(temp_vec,temp_vec);
    temp_vec = _mm256_hadd_ps(temp_vec,temp_vec);
    _mm256_store_ps(reduction_sub_force_y, temp_vec);

    fx[i] -= reduction_sub_force_x[0];
    fy[i] -= reduction_sub_force_y[0];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }

  #else
  printf("without simd\n");
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {// ベクトル化
      if(i != j) { // mask
        float rx = x[i] - x[j];
        float ry = y[i] - y[j]; // __mm256_sub_ps
        float r = std::sqrt(rx * rx + ry * ry);//_mm256_rsqrt_ps
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
  #endif

}
