#include "PatchMatchMain.cuh"
//#include <float.h>
#include "thrust/sort.h"

static __constant__ int2 chessboard_pair[20] = {
	{0,-1},{0,1},{0,-3},{0,3},{0,-5},{0,5},
	{1,0},{-1,0},{-3,0},{3,0},{5,0},{-5,0},
    {-1,2},{-1,-2},{-2,1},{-2,-1},
    {1,2},{1,-2},{2,1},{2,-1}
}; //constant memory of predefined chessboard pattern
static __constant__ int2 easy_pair[8] ={
	{0,-1},{0,1},
	{0,-5},{0,5},
    {1,0},{-1,0},
    {5,0},{-5,0}
};//constant memory of predefined chessboard pattern


__global__ void ComputeGray(uint8_t* color_img, uint8_t* gray_img, int height, int width)
{
	int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((idx_x < height) && (idx_y < width)) {
		uchar3 rgb = FETCH_UCHAR3(color_img[Get2dIdxRGB(idx_x, idx_y, height, width)]);
		uint8_t gray = uint8_t(float(rgb.x) * 0.299 + float(rgb.y) * 0.587 + float(rgb.z) * 0.114);
		gray_img[Get2dIdx(idx_x, idx_y, height, width)] = gray;
	}
}

__global__ void ComputeGrayGradient(uint8_t* gray_img, PMGradient* gradient, int height, int width, PMOption option)
{
    #define GRAY_GRAD_DIM 16
    #define BOUNDARY 1
	//padding with one
	__shared__ uint8_t smem[GRAY_GRAD_DIM+BOUNDARY*2][GRAY_GRAD_DIM+BOUNDARY*2];
	int idx_x = threadIdx.x+blockDim.x*blockIdx.x;
	int idx_y = threadIdx.y+blockDim.y*blockIdx.y;
	int thread_idx = threadIdx.x;
	int thread_idy = threadIdx.y;
	//mapping thread idx to original thread first
	bool is_valid= (idx_x - BOUNDARY < height) && (idx_y - BOUNDARY < width) && (idx_x - BOUNDARY >= 0) && (idx_y - BOUNDARY >= 0);
	//load the first block (top-left corner)
	smem[thread_idx][thread_idy] = is_valid ? gray_img[Get2dIdx(idx_x - BOUNDARY, idx_y - BOUNDARY, height, width)] : 0;

	//load the second block bottom-left corner
	is_valid = (idx_x + BOUNDARY < height) && (idx_x + BOUNDARY >= 0) && (idx_y - BOUNDARY >= 0) && (idx_y - BOUNDARY < width);
	if (thread_idx >= (GRAY_GRAD_DIM - BOUNDARY*2))
	{
		smem[thread_idx + BOUNDARY*2][thread_idy] = is_valid ? gray_img[Get2dIdx(idx_x + BOUNDARY, idx_y - BOUNDARY, height, width)] : 0;

	}
	//__syncthreads();
	//load the third block top-right corner
	is_valid = (idx_x - BOUNDARY >= 0) && (idx_x - BOUNDARY < height) && (idx_y + BOUNDARY < width) && (idx_y + BOUNDARY >= 0);
	if (thread_idy >= (GRAY_GRAD_DIM - 2*BOUNDARY))
	{
		smem[thread_idx][thread_idy + BOUNDARY*2] = is_valid ? gray_img[Get2dIdx(idx_x - BOUNDARY, idx_y + BOUNDARY, height, width)] : 0;
	}
	//__syncthreads();
	//load the fourth block bottom-right corner
	is_valid = (idx_x + BOUNDARY >= 0) && (idx_x + BOUNDARY < height) && (idx_y + BOUNDARY >= 0) && (idx_y + BOUNDARY < width);
	if (thread_idx >= (GRAY_GRAD_DIM - 2*BOUNDARY) && thread_idy >= (GRAY_GRAD_DIM - 2*BOUNDARY)) 
	{
		smem[thread_idx + BOUNDARY*2][thread_idy + BOUNDARY*2] = is_valid ? gray_img[Get2dIdx(idx_x + BOUNDARY, idx_y + BOUNDARY, height, width)] : 0;
	}
	__syncthreads();
	//feasible index of halo area
	if (idx_x >= BOUNDARY && idx_x < height - BOUNDARY && idx_y>=BOUNDARY && idx_y< width-BOUNDARY)
	{
		int i = thread_idx + BOUNDARY;
		int j = thread_idy + BOUNDARY; 
		//get the cooresponding shared index center location
		const int grad_x = -1 * int(smem[i - 1][j - 1]) +
			1 * int(smem[i - 1][j + 1])
			- 2 * int(smem[i][j - 1]) +
			2 * int(smem[i][j + 1]) +
			-1 * int(smem[i + 1][j - 1]) +
			1 * int(smem[i + 1][j + 1]);
		const int grad_y = -1 * int(smem[i - 1][j - 1])
			- 2 * int(smem[i - 1][j]) +
			-1 * int(smem[i - 1][j + 1]) +
			1 * int(smem[i + 1][j - 1]) +
			2 * int(smem[i + 1][j]) +
			1 * int(smem[i + 1][j + 1]);
		PMGradient grad = { grad_x / 8,grad_y / 8 };
		gradient[Get2dIdx(idx_x, idx_y, height, width)] = grad;
	}
	//else is zero position 
	//boundary condition
	else if (idx_x==0 && idx_x<height && idx_y>=0 && idx_y<width)
	{
		PMGradient grad = { 0,0 };
		gradient[Get2dIdx(idx_x, idx_y, height, width)] = grad;
	}
}

__global__ void InitRandState(curandState* rand_state, int height, int width,unsigned long long seed)
{
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = idx_x * width + idx_y;
	if (idx_x < height && idx_y < width)
	{   
		
		curand_init(idx, idx, 0, &rand_state[idx]);
	}
}

__global__ void InitRandomPlane(DisparityPlane* planes,  curandState* rand_state, int height, int width, int view,PMOption option)
{

	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int min_disp = option.disp_min;
	int max_disp = option.disp_max;
	float sign = view == 0 ? 1.0f : -1.0f;
	if (idx_x < height && idx_y < width)
	{   
		int idx = idx_x * width + idx_y;
		float disp_range = max_disp - min_disp;
		//generate random disparity e.g. (0 to 128)
		float random_d_val;
		//generate random normal (-1.0 to 1.0)
		float normal[3] = { 0.0f,0.0f,0.0f };
		float3 _normal;
		curandState localState = rand_state[idx];
		random_d_val= sign * (min_disp + disp_range * curand_uniform(&localState)); //generate 0.0 to 1.0
		for (int i = 0; i < 3; i++)
		{
				float np_rdval = curand_uniform(&localState);
				float np_sign = np_rdval > 0.5 ? 1.0f : -1.0f;
				float nval = np_sign * curand_uniform(&localState);  //XORWOW generator not generate 0.0,but include 1.0, so no need to check
				normal[i] = nval;
		}
		_normal.x = normal[0];
		_normal.y = normal[1];
		_normal.z = normal[2];
		_normal = Normalize(_normal);
		bool good_init=valid_init(idx_y,idx_x,_normal,random_d_val);
		if (!good_init)
			printf("[Plane Init]Not good init\n");
		DisparityPlane plane = DisparityPlane(idx_y, idx_x, _normal, random_d_val);
		planes[Get2dIdx(idx_x, idx_y, height, width)] = plane;
		rand_state[idx] = localState;
	}
}

__global__ void CostInit(float* costs,DisparityPlane* planes, uint8_t** views, PMGradient** grads,  int view, PMOption option)
{
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int height = option.height;
	int width = option.width;
	if (idx_x < height && idx_y < width)
	{   
		auto plane = planes[Get2dIdx(idx_x, idx_y, height, width)];
		float cost =  PatchMatchCost(plane, views, grads, idx_x, idx_y, view, option);
		costs[Get2dIdx(idx_x, idx_y, height, width)] = cost;
	}
}

__device__ float DisSimilarity(uchar3 color_p, PMGradient grad_p, int row, int col, float d, int view, uint8_t* views, PMGradient* grads, PMOption option)
{
	float col_r = float(col) - d;
	int height = option.height;
	int width = option.width;
	float alpha = option.alpha;
	float tau_color = option.tau_col;
	float tau_grad = option.tau_grad;
	if (isinf(col_r) || isnan(col_r)) {
		printf("[DIsSimialirtity]Not valid COL_R and d: %f,%f\n", col_r,d);
		return (1.0f - alpha) * tau_color + alpha * tau_grad;
	}
	if (col_r<0 || col_r>width - 1)
		return (1.0f - alpha) * tau_color + alpha * tau_grad;
	float3 color_q = get_interpolate_color(views, row, col_r, height, width);
	float dc = fmin(fabs(float(color_p.x) - float(color_q.x))
		   + fabs(float(color_p.y) - float(color_q.y)) 
		   +fabs(float(color_p.z) - float(color_q.z)), tau_color);
	float2 grad_q = get_interpolate_gradient(grads, row, col_r, height, width);
	float dg = fmin(fabs(float(grad_p._x) - float(grad_q.x)) + fabs(float(grad_p._y) - float(grad_q.y)), tau_grad);
	return (1.0f - alpha) * dc + alpha * dg;
}

__global__ void RowIteration(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, curandStateXORWOW_t* rand_state, int view, int cur_row, int direction, PMOption option)
{
	int row_idx = cur_row;
	int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
	int height = option.height;
	int width = option.width;
	if (col_idx < width) {
		DisparityPlane& plane_p = planes[view][Get2dIdx(row_idx, col_idx, height, width)];
		float& cost_p = costs[view][Get2dIdx(row_idx, col_idx, height, width)];
		int offsets[3] = { -1,0,1 };
		for (int i = 0; i < 3; i++) {
			int coord_x = row_idx - direction;
			int coord_y = col_idx + offsets[i];
			if (coord_x >= 0 && coord_x < height && coord_y >= 0 && coord_y < width)
			{
				DisparityPlane plane = planes[view][Get2dIdx(coord_x, coord_y, height, width)];
				//get a patch match cost of a different plane with current x and y
				float cost = PatchMatchCost(plane, views, grads, row_idx, col_idx, view, option);
				if (cost < cost_p) {
					plane_p = plane;
					cost_p = cost;
				}
			}
		}
		int max_disp;
		int min_disp;
		if (view == 0)
		{
			min_disp = option.disp_min;
			max_disp = option.disp_max;
		}
		else if (view == 1) {
			min_disp = -1 * option.disp_max;
			max_disp = option.disp_min;
		}
		float d_p = plane_p.ToDisparity(col_idx, row_idx);
		int coord = row_idx * width + col_idx;
		float3 norm_p = plane_p.ToNormal();
		float disp_update = (max_disp - min_disp) / 2.0f;
		float norm_update = 1.0f;
		float stop_thres = 0.25f; //0.1f is 10 times for disp
		curandState localState = rand_state[coord];
		while (disp_update > stop_thres) {
			float np_rdval = curand_uniform(&localState);
			float np_sign = np_rdval > 0.5 ? 1.0f : -1.0f;
			float disp_rd = np_sign * curand_uniform(&localState) * disp_update;
			float d_p_new = d_p + disp_rd;
			if (d_p_new<min_disp || d_p_new>max_disp)
			{
				disp_update /= 2;
				norm_update /= 2;
				continue;
			}
			//float3 norm_rd;
			float normal[3] = { 0.0f,0.0f,0.0f };
#pragma unroll
			for (int i = 0; i < 3; i++)
			{
				np_rdval = curand_uniform(&localState);
				np_sign = np_rdval > 0.5 ? 1.0f : -1.0f;
				float nval = np_sign * curand_uniform(&localState);  //XORWOW generator not generate 0.0,but include 1.0, so no need to check
				normal[i] = nval;
			}
			//norm_rd = { normal[0],normal[1],normal[2] };
			float3 norm_p_new = { normal[0] + norm_p.x,normal[1] + norm_p.y,normal[2] + norm_p.z };
			norm_p_new = Normalize(norm_p_new);
			if (!valid_init(col_idx, row_idx, norm_p_new, d_p_new)) {
				printf("[PlaneREfinement] Not good change\n");
				disp_update /= 2;
				norm_update /= 2;
				continue;
			}
			auto plane_new = DisparityPlane(col_idx, row_idx, norm_p_new, d_p_new);
			if (plane_new != plane_p) {
				const float cost = PatchMatchCost(plane_new, views, grads, row_idx, col_idx, view, option);
				if (cost < cost_p) {
					plane_p = plane_new;
					cost_p = cost;
					d_p = d_p_new;
					norm_p = norm_p_new;
				}
			}
			disp_update /= 2.0f;
			norm_update /= 2.0f;
		}
		rand_state[coord] = localState;
		//plane refinement ends
		//view propagation starts
		int coord_yr = lround(col_idx - d_p);
		if (coord_yr < 0 || coord_yr >= width || !isfinite(d_p))
			return;
		DisparityPlane& plane_q = planes[1 - view][Get2dIdx(row_idx, coord_yr, height, width)];
		auto& cost_q = costs[1 - view][Get2dIdx(row_idx, coord_yr, height, width)];
		DisparityPlane plane_p2q = ToAnotherView(plane_q);
		float cost = PatchMatchCost(plane_p2q, views, grads, row_idx, coord_yr, 1 - view, option);
		if (cost < cost_q)
		{
			plane_q = plane_p2q;
			cost_q = cost;
		}
		//view propagation ends
	}
}



//REMIND THAT X MEANS ROW AND Y MEANS COL IN CODE's CONTEXT
__device__ float PatchMatchCost(DisparityPlane plane, uint8_t** views, PMGradient** grads, int row, int col, int view, PMOption option)
{ 
	int min_disp = 0;
	int max_disp = 0;
	int height = option.height;
	int width = option.width;
	float gamma = option.gamma;
	if (view == 0)
	{
		min_disp = option.disp_min;
		max_disp = option.disp_max;
	}
	else if (view == 1) {
		min_disp = -1 * option.disp_max;
		max_disp = option.disp_min;
	}
	int half = option.win_size / 2;
	uint8_t* left_view = views[view];
	uint8_t* right_view = views[1 - view];
	PMGradient* left_grad = grads[view];
	PMGradient* right_grad = grads[1 - view];
	uchar3 color_p = FETCH_UCHAR3(left_view[Get2dIdxRGB(row, col, height, width)]);
	float cost = 0.0f;
	for (int r = -half; r <= half; r++) {
		int rowr = row + r;
		for (int c = -half; c <= half; c++) {
			int colc = col + c;
			if (colc<0 || colc>width - 1 || rowr<0 || rowr> height - 1)
				continue;
			//REMIND THAT X MEANS ROW AND Y MEANS COL IN THIS CONTEXT
			
			float d = plane.ToDisparity(colc, rowr);
			
			if (d<min_disp || d>max_disp) 
			{
				cost += COST_PUNISH;
				continue;
			}
			PMGradient grad_q = left_grad[Get2dIdx(rowr,colc,height,width)];
			uchar3 color_q = FETCH_UCHAR3(left_view[Get2dIdxRGB(rowr, colc, height, width)]);
			float w = Weight(color_p, color_q, gamma);
			float dis=DisSimilarity(color_q, grad_q, rowr, colc, d, view, right_view, right_grad,option);
			//if (row == 100 && col == 100)
			//	printf("cost init neighbors in view %d of %d,%d: w,dissim and disp:%f,%f,%f \n", view, row, col, w, dis, d);
			cost += w * dis;
		}
	}
	return cost;
}

__global__ void SpatialPropagation_Red(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option)
{   
	int height = option.height;
	int width = option.width;
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int2 coord = get_responsible_red_color_coord(idx_x, idx_y, height, width);
	if (coord.x != -1 && coord.y != -1) //valid coord
	{   
		assert(is_red(coord.x, coord.y, height, width));
		int coord_x = coord.x;
		int coord_y = coord.y;
		DisparityPlane& plane_p = planes[view][Get2dIdx(coord_x,coord_y,height,width)];
		float& cost_p = costs[view][Get2dIdx(coord_x, coord_y, height, width)];
#pragma unroll
		for (int i = 0; i < CHESSBOARD_PAIR_SIZE; i++) {

			int2 offset = chessboard_pair[i];
			int coord_x_offset = coord_x + offset.x;
			int coord_y_offset = coord_y + offset.y;
		
			bool in_range = (coord_x_offset >= 0) && (coord_x_offset < height) && (coord_y_offset>=0) && (coord_y_offset < width);
			if (in_range)
			{   
				assert(is_black(coord_x_offset, coord_y_offset, height, width));
				//get a different plane of chessboard pattern
				DisparityPlane plane = planes[view][Get2dIdx(coord_x_offset, coord_y_offset, height, width)];
				//get a patch match cost of a different plane with current x and y
				float cost = PatchMatchCost(plane, views, grads, coord_x, coord_y, view, option);
				if (cost < cost_p) {
					//planes[view][Get2dIdx(coord_x, coord_y, height, width)] = plane;
					//costs[view][Get2dIdx(coord_x, coord_y, height, width)] = cost;
					plane_p = plane;
					cost_p = cost;
				}
			}
		}
    //if(coord_x==100 && coord_y==100)
	//    printf("red pixel %d,%d has disparity: %f \n", coord_x, coord_y, plane_p.ToDisparity(coord_y,coord_x));
	}
}

__global__ void SpatialPropagation_Black(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option)
{
	
	int height = option.height;
	int width = option.width;
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int2 coord = get_responsible_black_color_coord(idx_x, idx_y, height, width);
	if (coord.x != -1 && coord.y != -1) //valid coord
	{   
		assert(is_black(coord.x, coord.y, height, width));
		int coord_x = coord.x;
		int coord_y = coord.y;
		DisparityPlane& plane_p = planes[view][Get2dIdx(coord_x, coord_y, height, width)];
		float& cost_p = costs[view][Get2dIdx(coord_x, coord_y, height, width)];
        #pragma unroll
		for (int i = 0; i < CHESSBOARD_PAIR_SIZE; i++) {
			int2 offset = chessboard_pair[i];
			int coord_x_offset = coord_x + offset.x;
			int coord_y_offset = coord_y + offset.y;
			bool in_range = (coord_x_offset >= 0) && (coord_x_offset < height) && (coord_y_offset>=0) && (coord_y_offset < width);
			if (in_range)
			{   
				assert(is_red(coord_x_offset, coord_y_offset, height, width));
				//get a different plane of chessboard pattern
				DisparityPlane plane = planes[view][Get2dIdx(coord_x_offset, coord_y_offset, height, width)];
				//get a patch match cost of a different plane with current x and y
				float cost = PatchMatchCost(plane, views, grads, coord_x, coord_y, view, option);
				if (cost < cost_p) {
					//planes[view][Get2dIdx(coord_x, coord_y, height, width)] = plane;
					//costs[view][Get2dIdx(coord_x, coord_y, height, width)] = cost;
					plane_p = plane;
					cost_p = cost;
				}
			}
		}
	}
}

__global__ void PlaneRefinement_Red(curandStateXORWOW_t* rand_state,float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option)
{
	int height = option.height;
	int width = option.width;
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int max_disp;
	int min_disp;
	if (view == 0)
	{
		min_disp = option.disp_min;
		max_disp = option.disp_max;
	}
	else if (view == 1) {
		min_disp = -1 * option.disp_max;
		max_disp = option.disp_min;
	}
	int2 coord = get_responsible_red_color_coord(idx_x, idx_y, height, width);
	if (coord.x != -1 && coord.y != -1) //valid coord
	{   
		int coord_x = coord.x;
		int coord_y = coord.y;
		int coord = coord_x * width + coord_y;
		DisparityPlane& plane_p = planes[view][Get2dIdx(coord_x, coord_y, height, width)];
		float& cost_p = costs[view][Get2dIdx(coord_x, coord_y, height, width)];
		float d_p = plane_p.ToDisparity(coord_y, coord_x);
		float3 norm_p = plane_p.ToNormal();
		float disp_update = (max_disp - min_disp) / 2.0f;
		float norm_update = 1.0f;
		float stop_thres = 0.1f; //0.1f is 10 times for disp
		curandState localState = rand_state[coord];
		while (disp_update > stop_thres) {
			float np_rdval = curand_uniform(&localState);
			float np_sign = np_rdval > 0.5 ? 1.0f : -1.0f;
			float disp_rd= np_sign * curand_uniform(&localState)* disp_update;
			float d_p_new = d_p + disp_rd;
			if (d_p_new<min_disp || d_p_new>max_disp)
			{
				disp_update /= 2.0f;
				norm_update /= 2.0f;
				continue;
			}
			//float3 norm_rd;
			float normal[3] = { 0.0f,0.0f,0.0f };
            #pragma unroll
			for (int i = 0; i < 3; i++)
			{
				np_rdval = curand_uniform(&localState);
				np_sign = np_rdval > 0.5 ? 1.0f : -1.0f;
				float nval = np_sign * curand_uniform(&localState);  //XORWOW generator not generate 0.0,but include 1.0, so no need to check
				normal[i] = nval;
			}
			//norm_rd = { normal[0],normal[1],normal[2] };
			float3 norm_p_new = { normal[0] + norm_p.x,normal[1] + norm_p.y,normal[2] + norm_p.z };
			norm_p_new = Normalize(norm_p_new);
			if (!valid_init(coord_y, coord_x, norm_p_new, d_p_new)) {
				printf("[PlaneREfinement] Not good change\n");
				continue;
			}
			auto plane_new = DisparityPlane(coord_y, coord_x, norm_p_new, d_p_new);
			if (plane_new != plane_p) {
				const float cost = PatchMatchCost(plane_new,views,grads,coord_x,coord_y,view,option);
				if (cost < cost_p) {
					plane_p = plane_new;
					cost_p = cost;
					d_p = d_p_new;
					norm_p = norm_p_new;
				}
			}
			disp_update /= 2.0f;
			norm_update /= 2.0f;
		}
		rand_state[coord] = localState;
		//if (idx_x == 100 && idx_y == 100)
		//{
		//	printf("red pixel %d,%d disparity: %f \n", coord_x, coord_y, d_p);
		//}
	}
}

__global__ void PlaneRefinement_Black(curandStateXORWOW_t* rand_state,float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option)
{
	int height = option.height;
	int width = option.width;
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int max_disp;
	int min_disp;
	if (view == 0)
	{
		min_disp = option.disp_min;
		max_disp = option.disp_max;
	}
	else if (view == 1) {
		min_disp = -1 * option.disp_max;
		max_disp = option.disp_min;
	}
	//int disp_range = max_disp - min_disp;
	int2 coord = get_responsible_black_color_coord(idx_x, idx_y, height, width);
	if (coord.x != -1 && coord.y != -1) //valid coord
	{
		int coord_x = coord.x;
		int coord_y = coord.y;
		int coord = coord_x * width + coord_y;
		DisparityPlane& plane_p = planes[view][Get2dIdx(coord_x, coord_y, height, width)];
		float& cost_p = costs[view][Get2dIdx(coord_x, coord_y, height, width)];
		float d_p = plane_p.ToDisparity(coord_y, coord_x);
		float3 norm_p = plane_p.ToNormal();
		float disp_update = (max_disp - min_disp) / 2.0f;
		float norm_update = 1.0f;
		float stop_thres = 0.25f; //0.1f is 10 times for disp
		curandState localState = rand_state[coord];
		while (disp_update > stop_thres) {
			float np_rdval = curand_uniform(&localState);
			float np_sign = np_rdval > 0.5 ? 1.0f : -1.0f;
			float disp_rd = np_sign * curand_uniform(&localState) * disp_update;
			float d_p_new = d_p + disp_rd;
			if (d_p_new<min_disp || d_p_new>max_disp)
			{
				disp_update /= 2;
				norm_update /= 2;
				continue;
			}
			//float3 norm_rd;
			float normal[3] = { 0.0f,0.0f,0.0f };
#pragma unroll
			for (int i = 0; i < 3; i++)
			{
				np_rdval = curand_uniform(&localState);
				np_sign = np_rdval > 0.5 ? 1.0f : -1.0f;
				float nval = np_sign * curand_uniform(&localState);  //XORWOW generator not generate 0.0,but include 1.0, so no need to check
				normal[i] = nval;
			}
			//norm_rd = { normal[0],normal[1],normal[2] };
			float3 norm_p_new = { normal[0] + norm_p.x,normal[1] + norm_p.y,normal[2] + norm_p.z };
			norm_p_new = Normalize(norm_p_new);
			if (!valid_init(coord_y, coord_x, norm_p_new, d_p_new)) {
				printf("[PlaneREfinement] Not good change\n");
				disp_update /= 2;
				norm_update /= 2;
				continue;
			}
			auto plane_new = DisparityPlane(coord_y, coord_x, norm_p_new, d_p_new);
			if (plane_new != plane_p) {
				const float cost = PatchMatchCost(plane_new, views, grads, coord_x, coord_y, view, option);
				if (cost < cost_p) {
					plane_p = plane_new;
					cost_p = cost;
					d_p = d_p_new;
					norm_p = norm_p_new;
				}
			}
			disp_update /= 2.0f;
			norm_update /= 2.0f;
		}
		rand_state[coord] = localState;
	}
}

__global__ void ViewPropagation_Red(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option)
{
	int height = option.height;
	int width = option.width;
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int2 coord = get_responsible_red_color_coord(idx_x, idx_y, height, width);
	if (coord.x != -1 && coord.y != -1) //valid coord
	{
		int coord_x = coord.x;
		int coord_y = coord.y;
		DisparityPlane& plane_p = planes[view][Get2dIdx(coord_x, coord_y, height, width)];
		float& cost_p = costs[view][Get2dIdx(coord_x, coord_y, height, width)];
		float d_p = plane_p.ToDisparity(coord_y, coord_x);
		int coord_yr = lround(coord_y - d_p);
		if (coord_yr < 0 || coord_yr >= width || !isfinite(d_p))
			return;
		DisparityPlane& plane_q = planes[1 - view][Get2dIdx(coord_x,coord_yr,height,width)];
		auto& cost_q = costs[1 - view][Get2dIdx(coord_x, coord_yr, height, width)];
		DisparityPlane plane_p2q = ToAnotherView(plane_q);
		//float d_q = plane_p2q.ToDisparity(coord_y, coord_x);
		float cost= PatchMatchCost(plane_p2q, views, grads, coord_x, coord_yr, 1-view, option);
		if (cost < cost_q)
		{
			plane_q = plane_p2q;
			cost_q = cost;
		}

	}

}

__global__ void ViewPropagation_Black(float** costs, DisparityPlane** planes, uint8_t** views, PMGradient** grads, int view, PMOption option)
{
	int height = option.height;
	int width = option.width;
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	int2 coord = get_responsible_black_color_coord(idx_x, idx_y, height, width);
	if (coord.x != -1 && coord.y != -1) //valid coord
	{
		int coord_x = coord.x;
		int coord_y = coord.y;
		//int coord = coord_x * width + coord_y;
		DisparityPlane& plane_p = planes[view][Get2dIdx(coord_x, coord_y, height, width)];
		float& cost_p = costs[view][Get2dIdx(coord_x, coord_y, height, width)];
		float d_p = plane_p.ToDisparity(coord_y, coord_x);
		int coord_yr = lround(coord_y - d_p);
		if (coord_yr < 0 || coord_yr >= width || !isfinite(d_p))
			return;
		DisparityPlane& plane_q = planes[1 - view][Get2dIdx(coord_x, coord_yr, height, width)];
		auto& cost_q = costs[1 - view][Get2dIdx(coord_x, coord_yr, height, width)];
		DisparityPlane plane_p2q = ToAnotherView(plane_q);
		//float d_q = plane_p2q.ToDisparity(coord_y, coord_x);
		float cost = PatchMatchCost(plane_p2q, views, grads, coord_x, coord_yr, 1 - view, option);
		if (cost < cost_q)
		{
			plane_q = plane_p2q;
			cost_q = cost;
		}
	}
}

__global__ void RetrieveDisparity(float* disp, DisparityPlane* plane, int view, PMOption option)
{
	int height = option.height;
	int width = option.width;
	int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
	int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
	if (idx_x < height && idx_y < width)
	{
		DisparityPlane plane_p = plane[Get2dIdx(idx_x, idx_y, height, width)];
		float d = plane_p.ToDisparity(idx_y, idx_x);
		disp[Get2dIdx(idx_x, idx_y, height, width)] = d;
	}
}


__global__ void outlier_detection(float* disp_left, float* disp_right, float* disp_out, uint8_t* disp_mask, PMOption option)
{
	int width = option.width;
	int height = option.height;
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
	float threshold = 2.0;
	if (x_idx < height && y_idx < width)
	{
		float disp = disp_left[Get2dIdx(x_idx, y_idx, height, width)];
		float col_right = lround(y_idx - disp); //get the corresponding pixel on right disp map
		if (col_right >= 0 && col_right < width)
		{
				float disp_r = disp_right[Get2dIdx(x_idx, int(col_right), height, width)];
				if (fabs(disp_r + disp) > threshold)
				{       
					//printf("disp diff:%f \n", abs(disp_r - disp));
					disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = MISMATCHES;
					disp_out[Get2dIdx(x_idx, y_idx, height, width)] = INVALID_FLOAT;
				}
				else
				{   
					
					disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = NORMAL;
					disp_out[Get2dIdx(x_idx, y_idx, height, width)] = disp;
				}
		}
		else
		{
				disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = MISMATCHES;
				disp_out[Get2dIdx(x_idx, y_idx, height, width)] = INVALID_FLOAT;
		}		
	}
}

__global__ void interpolation(float* disp, float* disp_buffer, uint8_t* disp_mask,DisparityPlane* plane, PMOption option)
{
	int width = option.width;
	int height = option.height;
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
	if (x_idx < height && y_idx < width)
	{
		uint8_t mask = disp[Get2dIdx(x_idx, y_idx, height, width)];
		if (mask == NORMAL)
		{
			disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = disp[Get2dIdx(x_idx, y_idx, height, width)];
		}
		else {
			DisparityPlane plane_left;
			float disp_left;
			DisparityPlane plane_right;
			float disp_right;
			int left_idx = y_idx-1;
			int right_idx = y_idx + 1;
			while (left_idx >= 0)
			{   
				if (disp[Get2dIdx(x_idx, left_idx, height, width)] != INVALID_FLOAT)
				{
					plane_left = plane[Get2dIdx(x_idx, left_idx, height, width)];
					disp_left = plane_left.ToDisparity(left_idx, x_idx);
					break;
				}
				else {
					left_idx -= 1;
				}
			}
			while (right_idx < width)
			{
				if (disp[Get2dIdx(x_idx, right_idx, height, width)] != INVALID_FLOAT)
				{
					plane_right = plane[Get2dIdx(x_idx, right_idx, height, width)];
					disp_right = plane_right.ToDisparity(right_idx, x_idx);
					break;
				}
				else {
					right_idx += 1;
				}
			}
			if (plane_left.Empty() && plane_right.Empty())
			{
				disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = disp[Get2dIdx(x_idx, y_idx, height, width)];
			}
			else if (plane_left.Empty() && !plane_right.Empty())
			{
				disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = disp_right;
				disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = NORMAL;
			}
			else if (!plane_left.Empty() && plane_right.Empty())
			{
				disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = disp_left;//
				disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = NORMAL;
			}
			else {
				float d_r = disp_right;//
				//float d_r = plane_right.ToDisparity(y_idx, x_idx);
				float d_l = disp_left;//abs(plane_left.ToDisparity(y_idx, x_idx));
				//float d_l= plane_left.ToDisparity(y_idx, x_idx);
				disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = abs(d_l)<abs(d_r)? d_l : d_r;
				disp_mask[Get2dIdx(x_idx, y_idx, height, width)] = NORMAL;
			}
		}
	}
}

__global__ void median_filter(float* disp_left, float* disp_out, int height, int width)
{
#define wnd_size 3  //i only implement a 3x3 filter because it's no so necessary to have a dynamic setting of this function,
	//you can use template to speed up instead
	int radius = wnd_size / 2;
	int actual_size = 0; //the actual size should consider in median filter
	int wnd_idx = 0;
	//int size = wnd_size * wnd_size;
	int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
	int y_idx = blockDim.y * blockIdx.y + threadIdx.y;
	if (x_idx < height && y_idx < width)
	{
		float local_mem[wnd_size * wnd_size] = { 0 };
#pragma unroll
		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++)
			{
				int x_coord = x_idx + i;
				int y_coord = y_idx + j;
				bool out_boundary = (x_coord < 0) || (y_coord < 0) || (x_coord >= height) || (y_coord >= width);
				if (out_boundary)
				{
					local_mem[wnd_idx] = -100000.0; //very low to make it sort to first
					wnd_idx++;
				}
				else {
					local_mem[wnd_idx] = disp_left[Get2dIdx(x_coord, y_coord, height, width)];
					wnd_idx++;
					actual_size++;
				}

			}
		}
		float local_disp_val = local_mem[wnd_size * wnd_size / 2];

		//bubble sort
		for (int i = 0; i < wnd_size * wnd_size - 1; i++)
		{
			for (int j = 0; j < wnd_size * wnd_size - i - 1; j++)
			{
				if (local_mem[j] > local_mem[j + 1])
				{
					float temp = local_mem[j + 1];
					local_mem[j + 1] = local_mem[j];
					local_mem[j] = temp;
				}
			}
		}
		float* valid_mem = (float*)&local_mem[wnd_size * wnd_size - actual_size];
		float median = 0;
		if (actual_size == 0)
		{
			disp_out[Get2dIdx(x_idx, y_idx, height, width)] = local_disp_val;
		}
		else {
			median = valid_mem[actual_size / 2];
			disp_out[Get2dIdx(x_idx, y_idx, height, width)] = median;
		}

	}
}

__global__ void weight_median_filter(float* disp, float* disp_buffer, uint8_t* disp_mask, uint8_t* view,  PMOption option)
{
	int width = option.width;
	int height = option.height;
	float gamma = option.gamma;
	int win_size = option.win_size;
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
	const int win_size2 = win_size / 2;
	if (x_idx < height && y_idx < width)
	{
		uint8_t mask = disp[Get2dIdx(x_idx, y_idx, height, width)];
		if (mask == NORMAL)
		{
			disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = disp[Get2dIdx(x_idx, y_idx, height, width)];
		}
		else {

			float total_w = 0.0f;
			int disp_idx = 0;
			float2* disps = (float2*)malloc(sizeof(float2) * (win_size+1) * (win_size+1));
			assert(disps != NULL);
			uchar3 col_p = FETCH_UCHAR3(view[Get2dIdxRGB(x_idx, y_idx, height, width)]);
			
			for (int r = -win_size2; r <= win_size2; r++) {
				for (int c = -win_size2; c <= win_size2; c++) {
					int xr = x_idx + r;
					int yc = y_idx + c;
					if(xr < 0 || xr >= height || yc < 0 || yc >= width)
						continue;
					float d=disp[Get2dIdx(xr, yc, height, width)];
					if (d == INVALID_FLOAT)
						continue;
					uchar3 col_q = FETCH_UCHAR3(view[Get2dIdxRGB(xr, yc, height, width)]);
					
					auto dc = abs(col_p.x - col_q.x) + abs(col_p.y - col_q.y) +
						abs(col_p.z - col_q.z);
					
					float w_temp = exp(-dc / gamma);
					total_w += w_temp;
					float2 temp_disp;
					temp_disp.x=d;
					temp_disp.y=w_temp;
					disps[disp_idx] = temp_disp;
					disp_idx += 1;
					
				}
			}
			
			
			float median_w = total_w / 2;
			float w = 0.0f;
			thrust::sort(thrust::seq,disps, disps + disp_idx, Float2Cmp());
			if (disp_idx >= 0)
			{
				for (int i = 0; i < disp_idx; i++)
				{
					w += disps[i].y;
					if (w >= median_w) {
						disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = disps[i].x;
						break;
					}
				}
			}
			else {
				disp_buffer[Get2dIdx(x_idx, y_idx, height, width)] = disp[Get2dIdx(x_idx, y_idx, height, width)];
			}
			
			free(disps);
		}
	}
}

