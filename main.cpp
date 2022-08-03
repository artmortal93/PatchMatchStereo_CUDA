
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "PatchMatchWrapper.cuh"
#include <filesystem>
using namespace std;
using namespace std::filesystem;
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <algorithm>
#include "stb_image_write.h"

// int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);


void write_plane(std::string name, DisparityPlane* planes, int width, int height) {
    std::ofstream myfile;
    myfile.open(name);
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            myfile << std::to_string(i) << " " << std::to_string(j) << " ";
            myfile << std::setprecision(8)<<planes[i * width + j].p.x <<" ";
            myfile << std::setprecision(8)<<planes[i * width + j].p.y <<" ";
            myfile << std::setprecision(8)<<planes[i * width + j].p.z <<" ";
            myfile << std::endl;
        }
    }



}

void write_disparity(std::string filename, float* disps, int _height, int _width, int min_disparity, int max_disparity)
{
    uint8_t* vis_data = new uint8_t[size_t(_height) * _width];
    float min_disp = float(_width), max_disp = -float(_width);
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            float disp = abs(disps[i * _width + j]);
            if (disp != INVALID_FLOAT) {
                min_disp = std::fmin(min_disp, disp);
                max_disp = std::fmax(max_disp, disp);
                
            }
        }
    }
    std::cout << "write out disparity, min disp is:" << min_disp << " max disp is" << max_disp << std::endl;
    for (int i = 0; i < _height; i++) {
        for (int j = 0; j < _width; j++) {
            const float disp = abs(disps[i * _width + j]);
            if (disp == INVALID_FLOAT) {
                vis_data[i * _width + j] = 0;
            }
            else {
                vis_data[i * _width + j] = static_cast<uint8_t>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }
    //write w,h,c
    int result = stbi_write_png(filename.c_str(), _width, _height, 1, (void*)vis_data,0);
    delete[] vis_data;
}

std::tuple<std::string, std::string, std::string> retrieve_middlebury_old(int idx) {
    std::vector<std::string> name = { "Aloe", "Baby1", "Baby2", "Baby3", "Bowling1",
                                        "Bowling2", "Cloth1", "Cloth2", "Cloth3", "Cloth4",
                                        "Flowerpots","Lampshade1", "Lampshade2", "Midd1", "Midd2",
                                        "Monopoly","Plastic", "Rocks1", "Rocks2", "Wood1",
                                        "Wood2" };
    auto imgdir_path = current_path() / "dataset";
    auto left_img_path = imgdir_path / name[idx] / "view1.png"; //disp1,disp5 in directory are ground true
    auto right_img_path = imgdir_path / name[idx] / "view5.png";
    return std::make_tuple(left_img_path.string(), right_img_path.string(), name[idx]);
}


std::tuple<std::string, std::string,std::string> retrieve_whu(std::string dir_index,std::string index) {

    auto imgdir_path = current_path() / "WHU_stereo_dataset"/"WHU_stereo_dataset";
    auto left_img_path = imgdir_path / "train" / dir_index / "Left" / index; //disp1,disp5 in directory are ground true
    auto right_img_path = imgdir_path /"train" /dir_index/ "Right"/index;
  
    return std::make_tuple(left_img_path.string(), right_img_path.string(),index);
}



int main()
{
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        
    }
    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL * 1024); //1GB
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda set heap size failed!");
        return 1;
    }



    //auto name_tuple = retrieve_middlebury_old(16);
    auto name_tuple = retrieve_whu("012_98", "012008.png");
    std::string left_img_path_str = std::get<0>(name_tuple);
    
    std::string right_img_path_str = std::get<1>(name_tuple);
    std::string img_name= std::get<2>(name_tuple);
    std::replace(left_img_path_str.begin(), left_img_path_str.end(), '\\', '\/');
    std::replace(right_img_path_str.begin(), right_img_path_str.end(), '\\', '\/');
    std::cout << left_img_path_str << std::endl;
    int l_width, l_height, l_bpp;
    int r_width, r_height, r_bpp;
    uint8_t* image_left = stbi_load(left_img_path_str.c_str(), &l_width, &l_height, &l_bpp, 3);
    std::cout << "image width: " << l_width << "image height: " << l_height << "bpp: " << l_bpp << std::endl;
    uint8_t* image_right = stbi_load(right_img_path_str.c_str(), &r_width, &r_height, &r_bpp, 3);
    std::cout << "image width: " << r_width << "image height: " << r_height << "bpp: " << r_bpp << std::endl;
    PMOption option;
    option.width = l_width;
    option.height = l_height;
    PatchMatchWrapper* compute_wrapper=new PatchMatchWrapper(option);
  
    compute_wrapper->Init();
    compute_wrapper->SetSourceImgs(image_left, image_right);
    compute_wrapper->Compute(8);   
    float* image_disp_left = compute_wrapper->RetrieveLeft();
    auto save_name = img_name + "pm_disp_left.png";
    write_disparity(save_name, image_disp_left, option.height, option.width, option.disp_min, option.disp_max);

    delete compute_wrapper;
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}

