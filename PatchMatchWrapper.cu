#include "PatchMatchWrapper.cuh"
#include "PatchMatchMain.cuh"
#include <stdio.h>
#include <time.h>

inline void gpuAssert(cudaError_t code, char* file, int line, bool Abort = true)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (Abort) exit(code);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


void PatchMatchWrapper::CleanUpMemory()
{
	FreeCudaResource();
}

void PatchMatchWrapper::AllocateCudaResource()
{
	int height = this->height;
	int width = this->width;
	gpuErrchk(cudaMalloc((void**)&img_left_rgb_d, (size_t)height * width * 3 * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&img_right_rgb_d, (size_t)height * width * 3 * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&img_left_gray_d, (size_t)height * width * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&img_right_gray_d, (size_t)height * width * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&disp_left_d, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&disp_right_d, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&cost_left_d, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&cost_right_d, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&grad_left_d, (size_t)height * width * sizeof(PMGradient)));
	gpuErrchk(cudaMalloc((void**)&grad_right_d, (size_t)height * width * sizeof(PMGradient)));
	gpuErrchk(cudaMalloc((void**)&plane_left_d, (size_t)height * width * sizeof(DisparityPlane)));
	gpuErrchk(cudaMalloc((void**)&plane_right_d, (size_t)height * width * sizeof(DisparityPlane)));
	gpuErrchk(cudaMalloc((void**)&disp_mask_left_d, (size_t)height * width * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&disp_mask_right_d, (size_t)height * width * sizeof(uint8_t)));
	gpuErrchk(cudaMalloc((void**)&disp_buffer_left_d, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&disp_buffer_right_d, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&rand_states, (size_t)height * width * sizeof(curandState_t)));

	gpuErrchk(cudaMalloc(&views,sizeof(uint8_t*)*2));
	gpuErrchk(cudaMalloc(&costs, sizeof(float*) * 2));
	gpuErrchk(cudaMalloc(&planes, sizeof(DisparityPlane*)*2));
	gpuErrchk(cudaMalloc(&disps, sizeof(float*) * 2));
	gpuErrchk(cudaMalloc(&grads, sizeof(PMGradient*) * 2));
     
	gpuErrchk(cudaMemcpy(&views[0], &img_left_rgb_d, sizeof(uint8_t*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&views[1], &img_right_rgb_d, sizeof(uint8_t*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&costs[0], &cost_left_d, sizeof(float*),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&costs[1], &cost_right_d, sizeof(float*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&planes[0], &plane_left_d, sizeof(DisparityPlane*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&planes[1], &plane_right_d, sizeof(DisparityPlane*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&disps[0], &disp_left_d, sizeof(float*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&disps[1], &disp_right_d, sizeof(float*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&grads[0], &grad_left_d, sizeof(PMGradient*), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(&grads[1], &grad_right_d, sizeof(PMGradient*), cudaMemcpyHostToDevice));
	streams = (cudaStream_t*)malloc(2 * sizeof(cudaStream_t));
	gpuErrchk(cudaStreamCreate(&streams[0]));
	gpuErrchk(cudaStreamCreate(&streams[1]));
	disp_left_h = (float*)malloc((size_t)width * height * sizeof(float));
	disp_right_h = (float*)malloc((size_t)width * height * sizeof(float));
	plane_left_h = (DisparityPlane*)malloc((size_t)width * height * sizeof(DisparityPlane));
	plane_right_h = (DisparityPlane*)malloc((size_t)width * height * sizeof(DisparityPlane));
	gpuErrchk(cudaMemset(grad_left_d, 0, (size_t)height * width * sizeof(PMGradient)));
	gpuErrchk(cudaMemset(grad_right_d, 0, (size_t)height * width * sizeof(PMGradient)));
	gpuErrchk(cudaMemset(disp_buffer_left_d, 0, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMemset(disp_buffer_right_d, 0, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMemset(disp_mask_left_d, 0, (size_t)height * width * sizeof(uint8_t)));
	gpuErrchk(cudaMemset(disp_mask_right_d, 0, (size_t)height * width * sizeof(uint8_t)));
	gpuErrchk(cudaPeekAtLastError());
	//only need to reset gradient because gradient is not cacualte for all img by default
	

}

void PatchMatchWrapper::FreeCudaResource()
{
	cudaFree(img_left_rgb_d);
	cudaFree(img_right_rgb_d);
	cudaFree(img_left_gray_d);
	cudaFree(img_right_gray_d);
	cudaFree(disp_left_d);
	cudaFree(disp_right_d);
	cudaFree(cost_left_d);
	cudaFree(cost_right_d);
	cudaFree(grad_left_d);
	cudaFree(grad_right_d);
	cudaFree(plane_left_d);
	cudaFree(plane_right_d);
	cudaFree(disp_mask_left_d);
	cudaFree(disp_mask_right_d);
	cudaFree(disp_buffer_left_d);
	cudaFree(disp_buffer_right_d);
	cudaFree(rand_states);
	gpuErrchk(cudaStreamDestroy(streams[0]));
	gpuErrchk(cudaStreamDestroy(streams[1]));
	free(streams);
	free(disp_left_h);
	free(disp_right_h);
	free(plane_left_h);
	free(plane_right_h);
}

void PatchMatchWrapper::SetSourceImgs(uint8_t* img_left, uint8_t* img_right)
{
	this->img_left_rgb_h = img_left;
	this->img_right_rgb_h = img_right;
	gpuErrchk(cudaMemcpy(this->img_left_rgb_d, this->img_left_rgb_h, (size_t)height * width * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(this->img_right_rgb_d, this->img_right_rgb_h, (size_t)height * width * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
}

void PatchMatchWrapper::Init()
{   
	if (first_time) {
		AllocateCudaResource();
		unsigned int blockDim = 32;
		dim3 grid = { unsigned int(this->height) / blockDim+1,unsigned int(this->width)/blockDim+1 ,1LLU };
		dim3 block = { blockDim,blockDim,1 };
		//seed by time
		srand(time(0));
		unsigned long long seed = rand();
		InitRandState <<< grid, block >>> (rand_states,this->height,this->width,seed);
		gpuErrchk(cudaPeekAtLastError());
		cudaDeviceSynchronize();
		first_time = false;
	}
	else {
		//do nothing 
	}
}

void PatchMatchWrapper::Reset(uint8_t* img_left, uint8_t* img_right)
{
	this->img_left_rgb_h = img_left;
	this->img_right_rgb_h = img_right;
	gpuErrchk(cudaMemcpy(this->img_left_rgb_d, this->img_left_rgb_h, (size_t)height * width * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(this->img_right_rgb_d, this->img_right_rgb_h, (size_t)height * width * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
	views[0] = img_left_rgb_d;
	views[1] = img_right_rgb_d;
	costs[0] = cost_left_d;
	costs[1] = cost_right_d;
	planes[0] = plane_left_d;
	planes[1] = plane_right_d;
	disps[0] = disp_left_d;
	disps[1] = disp_right_d;
	grads[0] = grad_left_d;
	grads[1] = grad_right_d;
	gpuErrchk(cudaMemset(disp_mask_left_d, 0, sizeof(uint8_t)*height*width));
	gpuErrchk(cudaMemset(disp_mask_right_d, 0, sizeof(uint8_t) * height * width));
	gpuErrchk(cudaMemset(grad_left_d, 0, (size_t)height * width * sizeof(PMGradient)));
	gpuErrchk(cudaMemset(grad_right_d, 0, (size_t)height * width * sizeof(PMGradient)));
	gpuErrchk(cudaMemset(disp_buffer_left_d, 0, (size_t)height * width * sizeof(float)));
	gpuErrchk(cudaMemset(disp_buffer_right_d, 0, (size_t)height * width * sizeof(float)));
	//reset some memory to zero to reuse
}

void PatchMatchWrapper::Compute(int iter)
{
	//use multiple stream to init
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	constexpr auto BLOCK_SIZE = 16;
	unsigned int block_dim_x = unsigned int(height) / BLOCK_SIZE + 1;
	unsigned int block_dim_y = unsigned int(width) / BLOCK_SIZE + 1;
	dim3 blockconfig = { BLOCK_SIZE,BLOCK_SIZE,1u };
	dim3 gridconfig = { block_dim_x,block_dim_y,1u };
	cudaEventRecord(start);
	ComputeGray <<<gridconfig, blockconfig, 0, streams[0] >>> (img_left_rgb_d,img_left_gray_d,height,width);
	ComputeGray <<<gridconfig, blockconfig, 1, streams[0] >>> (img_right_rgb_d,img_right_gray_d,height,width);
	ComputeGrayGradient <<<gridconfig, blockconfig, 0, streams[0]>>> (img_left_gray_d, grad_left_d, height,width,m_option);
	ComputeGrayGradient <<<gridconfig, blockconfig, 1, streams[0]>>> (img_right_gray_d, grad_right_d, height, width, m_option);
	for (int i = 0; i < 2; i++)
		cudaStreamSynchronize(streams[i]);
	//ok here 
	gpuErrchk(cudaPeekAtLastError());
	printf("finish compute gray and gray gradient\n");
	InitRandomPlane <<< gridconfig, blockconfig, 0, streams[0] >>> (plane_left_d,rand_states,height,width,0,m_option);
	cudaStreamSynchronize(streams[0]);
	InitRandomPlane <<< gridconfig, blockconfig, 0, streams[0] >>> (plane_right_d,rand_states,height,width,1, m_option);
	cudaStreamSynchronize(streams[0]);
	gpuErrchk(cudaPeekAtLastError());	
	printf("finish init random plane\n");
	//check error of costInit...
	CostInit <<<gridconfig, blockconfig, 0, streams[0] >>> (cost_left_d,plane_left_d,views,grads, 0, m_option);
	CostInit <<<gridconfig, blockconfig, 0, streams[0] >>> (cost_right_d,plane_right_d,views, grads, 1, m_option);
	for (int i = 0; i < 2; i++)
		cudaStreamSynchronize(streams[i]);
	gpuErrchk(cudaPeekAtLastError());

	printf("finish init cost\n");
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Ellaped time:%f ms\n", milliseconds);
	//init all the stream
	cudaEventRecord(start);
	if (m_option.mode == PM_MODE::RB) {
		for (int it = 0; it < iter; it++)
		{
			unsigned int worker_height = height;//(height+1) / 2;
			unsigned int worker_width = width;////(width+1) / 2;
			unsigned int worker_block_dim_x = worker_height / BLOCK_SIZE + 1;
			unsigned int worker_block_dim_y = worker_width / BLOCK_SIZE + 1;
			dim3 worker_blockconfig = { BLOCK_SIZE,BLOCK_SIZE,1 };
			dim3 worker_gridconfig = { worker_block_dim_x,worker_block_dim_y,1 };
			printf("iter:%d \n", it);

			for (int view = 0; view < 2; view++) {
				SpatialPropagation_Red << < worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (costs, planes, views, grads, view, m_option);
				cudaStreamSynchronize(streams[0]);
				gpuErrchk(cudaPeekAtLastError());
				PlaneRefinement_Red << < worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (rand_states, costs, planes, views, grads, view, m_option);
				cudaStreamSynchronize(streams[0]);
				gpuErrchk(cudaPeekAtLastError());
				ViewPropagation_Red << < worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (costs, planes, views, grads, view, m_option);
				cudaStreamSynchronize(streams[0]);
				SpatialPropagation_Black << < worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (costs, planes, views, grads, view, m_option);
				cudaStreamSynchronize(streams[0]);
				gpuErrchk(cudaPeekAtLastError());
				PlaneRefinement_Black << < worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (rand_states, costs, planes, views, grads, view, m_option);
				cudaStreamSynchronize(streams[0]);
				ViewPropagation_Black << < worker_gridconfig, worker_blockconfig, 0, streams[0] >> > (costs, planes, views, grads, view, m_option);
				cudaStreamSynchronize(streams[0]);

			}

			cudaStreamSynchronize(streams[0]);
		}
	}
	else {
		for (int it = 0; it < iter; it++)
		{
			unsigned int worker_height = height;
			unsigned int worker_width = width;
			unsigned int worker_block_dim_x = worker_width / BLOCK_SIZE + 1;
			dim3 worker_blockconfig = { BLOCK_SIZE,1,1 };
			dim3 worker_gridconfig = { worker_block_dim_x,1,1 };
			printf("iter:%d \n", it);
			for (int view = 0; view < 2; view++) {
				int direction = it % 2 == 0 ? 1 : -1;
				int start_row = (direction == 1) ? 1 : height - 2;
				int end_row = (direction == 1) ? height : -1;
				for (int r = start_row; r != end_row; r += direction)
				{
					RowIteration << <worker_gridconfig,worker_blockconfig, 0, streams[0] >> > (costs, planes, views, grads, rand_states, view, r, direction, m_option);
				}	
				cudaDeviceSynchronize();
			}


		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("stereo red black Ellaped time:%f ms\n", milliseconds);
	gpuErrchk(cudaPeekAtLastError());
	//*/
	PostProcess();
}

void PatchMatchWrapper::PostProcess()
{
	constexpr auto BLOCK_SIZE = 32;
	unsigned int block_dim_x = height / BLOCK_SIZE + 1;
	unsigned int block_dim_y = width / BLOCK_SIZE + 1;
	dim3 blockconfig = { BLOCK_SIZE,BLOCK_SIZE,1 };
	dim3 gridconfig = { block_dim_x,block_dim_y,1 };
	dim3 perpixelgridconfig = { (unsigned int)height,(unsigned int)width,1 };
	dim3 perpixelblockconfig = { 8,1,1 };
	RetrieveDisparity <<< gridconfig, blockconfig, 0, streams[0] >>> (disp_left_d, plane_left_d, 0, m_option);
	RetrieveDisparity <<< gridconfig, blockconfig, 0, streams[1] >>> (disp_right_d, plane_right_d, 1, m_option);
	for (int i = 0; i < 2; i++)
		cudaStreamSynchronize(streams[i]);
	cudaDeviceSynchronize();
	outlier_detection <<< gridconfig, blockconfig, 0, streams[0] >>> (disp_left_d, disp_right_d,disp_buffer_left_d,disp_mask_left_d, m_option);
	outlier_detection <<< gridconfig, blockconfig, 0, streams[0] >>> (disp_right_d, disp_left_d, disp_buffer_right_d, disp_mask_right_d, m_option);
	interpolation << < gridconfig, blockconfig, 0, streams[0] >> > (disp_buffer_left_d, disp_left_d, disp_mask_left_d, plane_left_d, m_option);
	gpuErrchk(cudaPeekAtLastError());
	weight_median_filter << < gridconfig, blockconfig, 0, streams[0] >> > (disp_left_d,disp_buffer_left_d,disp_mask_left_d,img_left_rgb_d,m_option);
	//gpuErrchk(cudaPeekAtLastError());
	cudaStreamSynchronize(streams[0]);
	cudaMemcpy(disp_left_h, disp_buffer_left_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(disp_right_h, disp_buffer_right_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(disp_left_h, disp_left_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	//cudaMemcpy(disp_right_h, disp_right_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
}

float* PatchMatchWrapper::RetrieveLeft()
{   
	return this->disp_left_h;
}

float* PatchMatchWrapper::RetrieveRight()
{   
	return this->disp_right_h;
}

DisparityPlane* PatchMatchWrapper::RetrievePlaneLeft()
{
	cudaMemcpy(plane_left_h, plane_left_d, sizeof(DisparityPlane) * width * height, cudaMemcpyDeviceToHost);
	return plane_left_h;
}

DisparityPlane* PatchMatchWrapper::RetrievePlaneRight()
{
	cudaMemcpy(plane_right_h, plane_right_d, sizeof(DisparityPlane) * width * height, cudaMemcpyDeviceToHost);
	return plane_right_h;
}
