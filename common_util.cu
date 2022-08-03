#include "common_util.cuh"
#include <float.h>


__device__ int Get3dIdx(int x, int y, int z, int xdim, int ydim, int zdim)
{
	int offset = x * ydim * zdim + y * zdim + z;
	return offset;
}

__device__ int Get3dIdxPitch(int x, int y, int z, int xdim, int ydim, int zdim, size_t pitch)
{
	//not implemented yet
	return 0;
}

__device__ int Get2dIdx(int x, int y, int xdim, int ydim)
{
	int offset = x * ydim + y;
	return offset;
}

__device__ int Get2dIdxRGB(int x, int y, int xdim, int ydim)
{
	int offset = x * ydim * 3 + y * 3;
	return offset;
}

__device__ int Get2dIdxPitch(int x, int y, int xdim, int ydim, size_t pitch)
{
	int offset = x * pitch + y;
	return offset;
}

__device__ int Get2dIdxPitchRGB(int x, int y, int xdim, int ydim, size_t pitch)
{
	//not implemented yet
	return 0;
}



 __device__ float3 Normalize(float3 vec)
{
	if (vec.x == 0.0f && vec.y == 0.0f && vec.z == 0.0f) {
		return vec;
	}
	else {
		float sq = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
		float sqf = sqrt(sq);
		vec.x /= sqf;
		vec.y /= sqf;
		vec.z /= sqf;
		return vec;
	}
}


 __device__ DisparityPlane ToAnotherView(DisparityPlane& plane)
 {
	 if (plane.p.x == 1.0f)
		 plane.p.x -= 0.001f; //avoid denom error with minus small delta value
	 float denom = 1.0f / (plane.p.x - 1.0f);
	 return DisparityPlane{ plane.p.x * denom, plane.p.y * denom, plane.p.z * denom };
 }

 __device__ float3 get_interpolate_color(uint8_t* img, int row, float col, int height, int width)
{   
	int col_cur = static_cast<int>(col);
	int col_nxt = col_cur + 1;
	if (col_nxt >= width)
	{
		col_nxt = col_cur;
	}
	float ofs = col - col_cur;
	uchar3 color1 = FETCH_UCHAR3(img[Get2dIdxRGB(row, col_cur, height, width)]);
	uchar3 color2 = color1;
	if(col_nxt!=col_cur)
		color2= FETCH_UCHAR3(img[Get2dIdxRGB(row, col_nxt, height, width)]);

	float3 interp_color;
	interp_color.x = (1 - ofs) * float(color1.x) + ofs * float(color2.x);
	interp_color.y = (1 - ofs) * float(color1.y) + ofs * float(color2.y);
	interp_color.z = (1 - ofs) * float(color1.z) + ofs * float(color2.z);
	return interp_color;
}

__device__ float2 get_interpolate_gradient(PMGradient* grad_left, int row, float col, int height, int width)
{
	int col_cur = static_cast<int>(col);
	int col_nxt = col_cur + 1;
	if (col_nxt >= width)
	{   
		//printf("strange gradient location: %d ,%d \n", col_cur, col_nxt);
		col_nxt = col_cur;
	}
	float ofs = col - col_cur;
	PMGradient grad1 = grad_left[Get2dIdx(row, col_cur, height, width)];

	PMGradient grad2 = grad1;
	if(col_nxt!=col_cur)
	   grad2=grad_left[Get2dIdx(row, col_nxt, height, width)];
	float2 interp_grad;
	interp_grad.x = (1 - ofs) * grad1._x + ofs * grad2._x;
	interp_grad.y = (1 - ofs) * grad1._y + ofs * grad2._y;
	return interp_grad;
}

///X not compact, compact in y
__device__ int2 get_responsible_red_color_coord(int XThreadIndex, int YThreadIndex, int height, int width)
{
	if (XThreadIndex >=height) {
		int2 t;
		t.x = -1;
		t.y = -1;
		return t;
	}
	else {
		int y_coord = XThreadIndex % 2 == 0 ? (YThreadIndex *2):(YThreadIndex * 2 + 1);
		if (y_coord >= width) {
			int2 t;
			t.x = -1;
			t.y = -1;
			return t;
		}
		else {
			int2 t;
			t.x = XThreadIndex;
			t.y = y_coord;
			return t;
		}
	}
}

__device__ int2 get_responsible_black_color_coord(int XThreadIndex, int YThreadIndex, int height, int width)
{
	if (XThreadIndex>=height)
		return { -1,-1 };
	else {
		int y_coord = XThreadIndex % 2 == 0 ? (YThreadIndex * 2+1) : (YThreadIndex * 2);
		if (y_coord >= width) {
			int2 t;
			t.x = -1;
			t.y = -1;
			return t;
		}
		else {
			int2 t;
			t.x = XThreadIndex;
			t.y = y_coord;
			return t;
		}
	}
}

__device__ bool is_red(int row, int col, int height, int width)
{
	if (row < 0 || row > height - 1 || col < 0 || col > width - 1)
		return false;
	if ((row + col) % 2 == 0)
		return true;
	else
		return false;

}

__device__ bool is_black(int row, int col, int height, int width)
{
	if (row < 0 || row > height - 1 || col < 0 || col > width - 1)
		return false;
	if ((row + col) % 2 == 0)
		return false;
	else
		return true;
}

__device__ float Weight(uchar3 p, uchar3 q, float gamma)
{
	int dc = abs(int(p.x) - int(q.x)) + abs(int(p.y) - int(q.y)) + abs(int(p.z) - int(q.z));
	float w = exp(float(-1.0 * float(dc) / gamma));
	return w;
}

__device__ int ColorDist(uchar3 c0, uchar3 c1)
{
	return max(abs(int(c0.x) - int(c1.x)), max(abs(int(c0.y) - int(c1.y)), abs(int(c0.z) - int(c1.z))));

}



DisparityPlane::DisparityPlane()
{
	p.x = 0.0;
	p.y = 0.0;
	p.z = 0.0;
};

DisparityPlane::DisparityPlane(float x, float y, float z)
{
	p.x = x;
	p.y = y;
	p.z = z;
}

DisparityPlane::DisparityPlane(float col, float row, float3 normal, float d)
{
	//check outside
	//normal.z must not be 0
	this->p.x = -1.0f * normal.x / normal.z;
	this->p.y = -1.0f * normal.y / normal.z;
	this->p.z = (normal.x * col + normal.y * row + normal.z * d) / normal.z;
	//this->nl.x = normal.x;
	//this->nl.y = normal.y;
	//this->nl.z = normal.z; //could sometimes gets to 0 with unknown reason
}

__device__ bool valid_init(const float x, const float y, const float3 normal, const float d)
{
	if (normal.z == 0.0)
		return false;
	float  _p0 = -1.0 * normal.x / normal.y;
	float  _p1 = -1.0 * normal.x / normal.y;
	float _p2 = (normal.x * x + normal.y * y + normal.z * d) / normal.z;
	float check[3] = { _p0,_p1,_p2 };
	for (int i = 0; i < 3; i++)
	{
		if (isinf(check[i]) || isnan(check[i]))
			return false;
	}
	return true;
}


__host__ __device__ float DisparityPlane::ToDisparity(int col, int row)
{
	float d = float(col) * this->p.x + float(row) * this->p.y + 1.0f * this->p.z;
	if (isinf(d) || isnan(d)) {
		printf("Not valid disp in %d,%d: %f\n",row,col, d);
		printf("Nov valid coefficient: %f %f %f\n", this->p.x, this->p.y, this->p.z);
	}
	return d;
}

__host__ __device__ bool DisparityPlane::Empty()
{
	return (this->p.x == 0.0f && this->p.y == 0.0f && this->p.z == 0.0f);
}

__host__ __device__ float3 DisparityPlane::ToNormal()
{
	float3 n = { this->p.x,this->p.y,-1.0f };
	n = Normalize(n);
	return n;
}

__host__ __device__ DisparityPlane& DisparityPlane::operator=(const DisparityPlane& other)
{
	if (this != &other) {
		this->p.x = other.p.x;
		this->p.y = other.p.y;
		this->p.z = other.p.z;
	}
	return *this;
}

__host__ __device__ DisparityPlane& DisparityPlane::operator=(DisparityPlane& other)
{
	if (this != &other) {
		this->p.x = other.p.x;
		this->p.y = other.p.y;
		this->p.z = other.p.z;
	}
	return *this;
}

__host__ __device__ bool DisparityPlane::operator!=(const DisparityPlane& v) const
{
	return (p.x != v.p.x) || (p.y != v.p.y) || (p.z != v.p.z);
}

__host__ __device__ bool DisparityPlane::operator==(const DisparityPlane& v) const
{
	return (p.x == v.p.x) && (p.y == v.p.y) && (p.z != v.p.z);
}



