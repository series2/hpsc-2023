// #include <iostream>
#include<cstdlib>
#include<cstdio>
#include<vector>
#include<chrono>
//#include "matplotlibcpp.h"
#include <immintrin.h>

/*
OpenMP 10
MPI 40
SIMD 20
OpenACC 10
CUDA 40
*/
__global__ void thread(float *b,int j,int i,float ryo,float dt,float dx,float dy,float** u float**v){
    b[j][i+threadIdx.x] = rho * (1 / dt *\
        ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[1][i] - v[1][i]) / (2 * dy)) -\
        ((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 2 * ((u[1][i] - u[1][i]) / (2 * dy) *\
        (v[j][i+1] - v[j][i-1]) / (2 * dx)) - ((v[1][i] - v[1][i]) / (2 * dy))* ((v[1][i] - v[1][i]) / (2 * dy)));
}

using namespace std;
//namespace plt = matplotlibcpp;
// typedef vector<float*> matrix;
int main(){
    const int nx = 42;
    const int ny = 42;
    int nt = 500;
    int nit = 50;
    float dx = 2. / (nx - 1);
    float dy = 2. / (ny - 1);
    float dt = 0.01;
    float rho = 1;
    float nu = 0.02;
    // vector<float> x(nx);
    // vector<float> y(ny);
    float x[nx];
    float y[ny];
    for (int i=0;i<nx;i++) x[i]=i*dx;
    for (int i=0;i<ny;i++) y[i]=i*dy;
    // matrix u(ny,vector<float>(nx,0));
    // matrix v(ny,vector<float>(nx,0));
    // matrix b(ny,vector<float>(nx,0));
    // matrix p(ny,vector<float>(nx,0));
    float u[ny][nx];
    float v[ny][nx];
    float b[ny][nx];
    float p[ny][nx];
    for (int j=0;j<ny;j++)for (int i=0;i<nx;i++) {u[j][i]=0;v[j][i]=0;b[j][i]=0;p[j][i]=0;}

    // matrix un(ny,vector<float>(nx,0));
    // matrix vn(ny,vector<float>(nx,0));
    // matrix pn(ny,vector<float>(nx,0));
    float un[ny][nx];
    float vn[ny][nx];
    float pn[ny][nx];
    for (int j=0;j<nu;j++)for (int i=0;i<nx;i++) {un[j][i]=0;vn[j][i]=0;pn[j][i]=0;}

    //X, Y = np.meshgrid(x, y)
    //matrix X(ny,vector<float>(nx));
    //for(int j=0;j<ny;j++) for(int i=0;i<nx;i++) X[j][i]=x[i];
    //matrix Y(ny,vector<float>(ny));
    //for(int j=0;j<ny;j++) for(int i=0;i<nx;i++) Y[j][i]=y[j];

    for(int n=0;n<nt;n++){
        auto tic = chrono::steady_clock::now();
        for(int j=1;j<ny-1;j++){
            int num=16;
            for(int i=1;i<nx-1;i+=num){
                cudaMallocManaged(&b[j][i],num*sizeof(float));
                thread<<1,num>>(&b[j][i],j,i,ryo, dt, dx, dy,u ,v);
                // b[j][i] = rho * (1 / dt *\
                //         ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[1][i] - v[1][i]) / (2 * dy)) -\
                //         ((u[j][i+1] - u[j][i-1]) / (2 * dx))*((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 2 * ((u[1][i] - u[1][i]) / (2 * dy) *\
                //         (v[j][i+1] - v[j][i-1]) / (2 * dx)) - ((v[1][i] - v[1][i]) / (2 * dy))* ((v[1][i] - v[1][i]) / (2 * dy)));
                cudaDeviceSyncronize();
                cudaFree(b)
            } 
        }
        for(int it=0;it<nit;it++){
            // pをバッファに保存
            for(int j=0;j<ny;j++){
                for(int i=0;i<nx;i++){
                    pn[j][i]=p[j][i];
                }
            }

            // ポアソン
#pragma omp parallel for // here
            for(int j=1;j<ny-1;j++){
                for(int i=1;i<nx-1;i++){
                    p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +\
                            dx*dx * (pn[1][i] + pn[1][i]) -\
                            b[j][i] * (dx*dx) * (dy*dy))\
                            / (2 * ((dx*dx)  +(dy*dy)));
                }
            }
            // 境界条件
            for(int j=1;j<ny-1;j++){
                p[j][nx-1]=p[j][nx-2];
                p[j][0]=p[j][1]; // 勾配0
            }
            for(int i=1;i<nx-1;i++){
                p[0][i]=p[1][i]; // 勾配0
                p[ny-1][i]=p[ny-2][i]; // 勾配0
            }
        }
        // uをバッファに保存
        for(int j=0;j<ny;j++){

            for(int i=0;i<nx;i++){
                vn[j][i]=v[j][i];
                un[j][i]=u[j][i];
            }
        }
        for(int j=1;j<ny-1;j++){
            for(int i=1;i<nx-1;i+=8){
                __m256 unvecji = _mm256_load_ps(&un[j][i]);
                __m256 unvecjm = _mm256_load_ps(&un[j][i-1]);
                __m256 unvecjp = _mm256_load_ps(&un[j][i+1]);
                __m256 unvec1i = _mm256_load_ps(&un[1][i]);
                __m256 pvecjm = _mm256_load_ps(&p[j][i+1]);
                __m256 pvecjp = _mm256_load_ps(&p[j][i-1]);

                __m256 vnvecji = _mm256_load_ps(&vn[j][i]);
                __m256 vnvecjm = _mm256_load_ps(&vn[j][i-1]);
                __m256 vnvecjp = _mm256_load_ps(&vn[j][i+1]);
                __m256 vnvec1i = _mm256_load_ps(&vn[1][i]);

                __m256 uvecji = _mm256_load_ps(&u[j][i]);
                __m256 vvecji = _mm256_load_ps(&v[j][i]);

                uvecji = unvecji - unvecji * dt / dx * (unvecji - unvecjm)\
                                - unvecji * dt / dy * (unvecji - unvec1i)\
                                - dt / (2 * rho * dx) * (pvecjp - pvecjm)\
                                + nu * dt / (dx*dx) * (unvecjp - 2 * unvecji + unvecjm)\
                                + nu * dt / (dy*dy) * (unvec1i - 2 * unvecji + unvec1i);
                vvecji = vnvecji - vnvecji * dt / dx * (vnvecji - vnvecjm)\
                                - vnvecji * dt / dy * (vnvecji - vnvec1i)\
                                - dt / (2 * rho * dx) * (pvecjp - pvecjm)\
                                + nu * dt / (dx*dx) * (vnvecjp - 2 * vnvecji + vnvecjm)\
                                + nu * dt / (dy*dy) * (vnvec1i - 2 * vnvecji + vnvec1i);
                // _mm256_store_ps(&u[i][j],uvecji);
                // _mm256_store_ps(&v[i][j],vvecji);
                 
            }
        }
        
        // 境界条件
#pragma omp parallel
        for(int j=1;j<ny-1;j++){
            u[j][nx-1]=0;
            u[j][0]=0;
            v[j][0]=0;
            v[j][nx-1]=0;
        }
#pragma omp parallel
        for(int i=1;i<nx-1;i++){
            u[0][i]=0;
            u[nx-1][i]=1;
            v[0][i]=0;
            v[nx-1][1]=0;
        }

        auto toc = chrono::steady_clock::now();
        double time = chrono::duration<double>(toc-tic).count();
        printf("step=%d: %lf s \n",n,time);

        //plt::contour(X, Y, p);
        //plt::quiver(X, Y, u, v);
        //plt::pause(0.01);
        //plt::clf();
    }
    //plt::show();
}

