#pragma once


#include <utility>
#include <stdint.h>
#include "common_util.cuh"
#include <curand_kernel.h>


class PatchMatchWrapper
{
public:
	PatchMatchWrapper(PMOption& option):
		m_option(option)
	{
		height = option.height;
		width = option.width;
	}

	~PatchMatchWrapper()
	{
		CleanUpMemory();
	}

public: 
	/*
	Only applied in first time allocation, for reuse the memory after first computation, use Reset()
	*/
	 void SetSourceImgs(uint8_t* img_left, uint8_t* img_right);
	//Init all the first time buffer
	 void Init();
	 void Reset(uint8_t* img_left, uint8_t* img_right);
	 void Compute(int iter);
	 void PostProcess();
	 float* RetrieveLeft();
	 float* RetrieveRight();
	 DisparityPlane* RetrievePlaneLeft();
	 DisparityPlane* RetrievePlaneRight();

//for debug purpose not setting protected
public:
	bool first_time = true;
	bool multistream = true;
	uint8_t* img_left_rgb_h = nullptr;
	uint8_t* img_right_rgb_h = nullptr;
	uint8_t* img_left_rgb_d = nullptr;
	uint8_t* img_right_rgb_d = nullptr;
	uint8_t* img_left_gray_d = nullptr;
	uint8_t* img_right_gray_d = nullptr;
	float* cost_left_d = nullptr;
	float* cost_right_d = nullptr;
	float* disp_left_d = nullptr;
	float* disp_right_d = nullptr;
	float* disp_left_h = nullptr;  //host mem
	float* disp_right_h = nullptr; //host mem
	PMGradient* grad_left_d = nullptr;
	PMGradient* grad_right_d = nullptr;
	DisparityPlane* plane_left_d = nullptr;
	DisparityPlane* plane_right_d = nullptr;
	DisparityPlane* plane_left_h = nullptr;
	DisparityPlane* plane_right_h = nullptr;
	//post processing memory
	uint8_t* disp_mask_left_d = nullptr;
	uint8_t* disp_mask_right_d = nullptr;
	float* disp_buffer_left_d = nullptr;
	float* disp_buffer_right_d = nullptr;
	cudaStream_t* streams;
	//a set of convinent collection set to operate in different view
	uint8_t** views;
	float** costs;
	DisparityPlane** planes;
	float** disps;
	PMGradient** grads;
	//generate 0.0f to 1.0f number(0.0 excluded 1.0 included)
	curandState_t* rand_states;
	
protected:
	int height;
	int width;
	PMOption m_option;
	//clean up the memory of the all the buffer to proper state in order to reuse the cuda memory(for not first time usage)
	void CleanUpMemory();
	//allocate first time use memory on device
	void AllocateCudaResource();
	void FreeCudaResource();


};