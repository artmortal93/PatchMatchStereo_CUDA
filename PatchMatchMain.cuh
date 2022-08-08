#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include "common_util.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "intellisense_cuda_intrinsics.h"
#include <math.h>
#include <curand.h>
#include <assert.h>


#define EASY_PAIR_SIZE 8
#define CHESSBOARD_PAIR_SIZE 20

__global__ void ComputeGray(uint8_t* color_img, uint8_t* gray_img, int height, int width);//finish


__global__ void ComputeGrayGradient(uint8_t* gray_img, PMGradient* gradient, int height, int width, PMOption option);


__global__ void InitRandState(curandState* rand_state,int height, int width,unsigned long long seed);


__global__ void InitRandomPlane(DisparityPlane* planes,curandState* rand_state, int height, int width, int view,PMOption option);


__global__ void CostInit(float* cost,DisparityPlane* plane, uint8_t** views, PMGradient** grads,  int view, PMOption option);


__device__ float PatchMatchCost(DisparityPlane plane, uint8_t** views, PMGradient** grads, int x, int y, int view, PMOption option);

__global__ void SpatialPropagation_Red(float** costs,DisparityPlane** planes, uint8_t** views, PMGradient** grads,int view, PMOption option);

__global__ void SpatialPropagation_Black(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option);

__global__ void PlaneRefinement_Red(curandStateXORWOW_t* rand_state,float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option);

__global__ void PlaneRefinement_Black(curandStateXORWOW_t* rand_state,float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option);

__global__ void ViewPropagation_Red(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option);

__global__ void ViewPropagation_Black(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option);

__global__ void RetrieveDisparity(float* disp, DisparityPlane* plane, int view, PMOption option);


//LRCheck
__global__ void outlier_detection(float* disp_left, float* disp_right, float* disp_out, uint8_t* disp_mask, PMOption option);

__global__ void interpolation(float* disp, float* disp_buffer, uint8_t* disp_mask, DisparityPlane* plane, PMOption option);

__global__ void median_filter(float* disp_left, float* disp_out, int height, int width);

__global__ void weight_median_filter(float* disp_left, float* disp_out, uint8_t* disp_mask, uint8_t* view,PMOption option);

//get gradient for
__device__ float DisSimilarity(uchar3 color_p, PMGradient grad_p, int row, int col, float d, int view, uint8_t* views, PMGradient* grads, PMOption option);


__global__ void RowIteration(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, curandStateXORWOW_t* rand_state,int view,int cur_row, int direction, PMOption option);