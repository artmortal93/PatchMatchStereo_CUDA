#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "intellisense_cuda_intrinsics.h"
#include <stdint.h>
#include <math.h>
#include <stdio.h>

#define INVALID_FLOAT (INFINITY)
#define COST_PUNISH 120.0f
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_UCHAR3(pointer) (reinterpret_cast<uchar3*>(&(pointer))[0])

#define NORMAL 0
#define OCCLUSIONS 2
#define MISMATCHES 1



__device__ int Get3dIdx(int x, int y, int z, int xdim, int ydim, int zdim);


__device__ int Get3dIdxPitch(int x, int y, int z, int xdim, int ydim, int zdim, size_t pitch);


__device__ int Get2dIdx(int x, int y, int xdim, int ydim);

__device__ int Get2dIdxRGB(int x, int y, int xdim, int ydim);


__device__ int Get2dIdxPitch(int x, int y, int xdim, int ydim, size_t pitch);

__device__ int Get2dIdxPitchRGB(int x, int y, int xdim, int ydim, size_t pitch);



__host__ __device__ float3 Normalize(float3 vec);


enum PM_COLOR {
    RED,
    BLACK,
    INVALID
};

enum PM_MODE {
    RB,
    ROW_SWEEP
};

struct PMOption
{
    int height=0;
    int width=0;
    float alpha = 0.0f;
    float gamma = 10.0f;
    float tau_grad = 2.0f;
    float tau_col = 10.0f;
    int disp_min = 0;
    int disp_max = 128;
    int win_size = 16;
    int patch_size = 35;
    int mode = PM_MODE::RB;
};

struct PMGradient {
    int _x = 0;
    int _y = 0;
};

struct Float2Cmp {
    __host__ __device__
        bool operator()(const float2& o1, const float2& o2) {
        return o1.x < o2.x;
    }
};

class DisparityPlane
{   
public:
    __host__ __device__ DisparityPlane();
    

    __host__ __device__ DisparityPlane(float x, float y, float z);
    
    //REMIND THAT WE CHANGE THE NOTATION TO Y AS COL AND X AS ROW
    __host__ __device__ DisparityPlane(float col, float row, float3 normal, float d);
  
    //REMIND THAT WE CHANGE THE NOTATION TO Y AS COL AND X AS ROW


    //REMIND THAT WE CHANGE THE NOTATION TO Y AS COL AND X AS ROW
    __host__ __device__ float ToDisparity(int col, int row);

    __host__ __device__ bool Empty();

    __host__ __device__ float3 ToNormal();

    __host__ __device__ DisparityPlane& operator=(const DisparityPlane& other);

    __host__ __device__ DisparityPlane& operator=(DisparityPlane& other);
   

    __host__ __device__ bool operator!=(const DisparityPlane& v) const;

    __host__ __device__ bool operator==(const DisparityPlane& v) const;
       
public:
    float3 p;
};

__device__ DisparityPlane ToAnotherView(DisparityPlane& plane);

__device__ float3 get_interpolate_color(uint8_t* img, int row, float col, int height, int width);

__device__ float2 get_interpolate_gradient(PMGradient* grad_left, int row, float col, int height, int width);

__device__ int2 get_responsible_red_color_coord(int XThreadIndex, int YThreadIndex, int height, int width);

__device__ int2 get_responsible_black_color_coord(int XThreadIndex, int YThreadIndex, int height, int width);

__device__ bool is_red(int row, int col, int height, int width);

__device__ bool is_black(int row, int col, int height, int width);

__device__ float Weight(uchar3 p, uchar3 q, float gamma);

__device__ int ColorDist(uchar3 c0, uchar3 c1);

__device__ bool valid_init(const float col, const float row, const float3 normal, const float d);

