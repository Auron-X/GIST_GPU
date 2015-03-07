
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2\opencv.hpp>
#include <iostream>
#include <cufft.h>

using namespace std;
using namespace cv;

# define PI           3.14159265358979323846
# define N			  256

cudaError_t cudaStatus;

bool TIMER_FLG = 1;

const int numOfScales = 4;
const int numOfOrient = 8;
const int numOfFilters = numOfScales*numOfOrient;
const int filterSizeX = N;
const int filterSizeY = N;	
const int width = N;
const int height = N;	

float *dev_G_data = 0; 

cufftHandle fft_plan = NULL, 
			ifft_plan = NULL;

__global__ void gaborCalcKernel(int numOfFilters, float* gaborParams, float* G)
{
    //Declaration
	int width = gridDim.x*blockDim.x;
	int height = gridDim.y*blockDim.y;
	int curX = blockIdx.x*blockDim.x+threadIdx.x;
	int curY = blockIdx.y*blockDim.y+threadIdx.y;
	int indx = curY*width+curX;
	int fx, fy;
	float fr1, t1, tr;

	//FFT Shift
	if (curX < width/2.0) fx = curX + width/2.0;
		else fx = curX - width/2.0;
	if (curY < height/2.0) fy = curY + height/2.0;
		else fy = curY - height/2.0;
	fx -= width/2.0;
	fy -= height/2.0;

	//Precalculate fr & t
	fr1 = sqrtf(fx*fx + fy*fy);
    t1 = atan2f(fy, fx);

	//Calculate Gabor Filter values
	for (int n=0; n<numOfFilters; n++)
	{

		tr = t1+gaborParams[n*4+3]; 
		if(tr < -PI) tr += 2.0f*PI;
		else if (tr > PI) tr -= 2.0f*PI;
			
		G[n*width*height+indx] = exp(-10.0f*gaborParams[n*4]*(fr1/height/gaborParams[n*4+1]-1)*
				(fr1/width/gaborParams[n*4+1]-1)-2.0f*gaborParams[n*4+2]*PI*tr*tr);
	}
}

__global__ void multiplyKernel(float* img_fft, float* G, float* res, int k)
{
	int width = gridDim.x*blockDim.x;
	int height = gridDim.y*blockDim.y;
	int curX = blockIdx.x*blockDim.x+threadIdx.x;
	int curY = blockIdx.y*blockDim.y+threadIdx.y;
	int indx = curY*width+curX;
	
	res[indx*2+0] = img_fft[indx*2+0]*G[k*width*height+indx];
	res[indx*2+1] = img_fft[indx*2+1]*G[k*width*height+indx];
}

float* generateGaborFiltersGPU()
{
    float *dev_gaborParams = 0;
	cudaEvent_t start, stop;
	cudaEvent_t full_start, full_stop;
	float elapsed_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&full_start);
	cudaEventCreate(&full_stop);

	cudaEventRecord(start);
	cudaEventSynchronize(start);
	
	//Generate Gabor Filter parameters
	float gaborParams[numOfFilters][4]; 
	for (int i=0; i<numOfScales; i++)
		for (int j=0; j<numOfOrient; j++)
		{
			gaborParams[i*numOfOrient+j][0] = 0.35;
			gaborParams[i*numOfOrient+j][1] = 0.3/powf(1.85,i);
			gaborParams[i*numOfOrient+j][2] = 16.0*numOfOrient*numOfOrient/32.0/32.0;
			gaborParams[i*numOfOrient+j][3] = PI/numOfOrient*j;			
		}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Gabor param time: " << elapsed_time << endl;

	cudaEventRecord(start);
	cudaEventSynchronize(start);
	//Gabor Calculations of CUDA
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)   
	cudaStatus = cudaMalloc((void**)&dev_gaborParams, numOfFilters*4*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_G_data, numOfFilters*width*height*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	

    // Copy input vectors from host memory to GPU buffers.
	
	cudaStatus = cudaMemcpy(dev_gaborParams, &gaborParams[0][0], numOfFilters*4*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Memcpy time: " << elapsed_time << endl;
    // Launch a kernel on the GPU with one thread for each element.
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	gaborCalcKernel <<< dim3(width/32, height/32), dim3(32, 32)>>> (numOfFilters, dev_gaborParams, dev_G_data);
	
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
   cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Gabor calc time: " << elapsed_time << endl;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
			
    // Copy output vector from GPU buffer to host memory.

	float *G_data = (float *) malloc(numOfFilters*width*height*sizeof(float));
    cudaStatus = cudaMemcpy(G_data, dev_G_data, numOfFilters*width*height*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	// Copy G data to Mat vector
	/*for (int n=0; n<numOfFilters; n++)
	{
		Mat G0(width, height, CV_32F);
		float* G_ptr = (float*)G0.data;
		memcpy(G_ptr, &G_data[n*width*height], width*height*sizeof(float));
		G.push_back(G0);
	}*/

Error:
    cudaFree(dev_gaborParams);
	//cudaFree(dev_G_data);
	return G_data;
}



float* calcGistGPU(Mat img, float* G)
{
	//Declarations
	cufftComplex *out_data, *dev_resComplex;
	float *dev_fft_img, *dev_res;
	cufftReal *in_data;
	float* result = (float *) malloc(numOfFilters*width*height*sizeof(float));

	cudaEvent_t start, stop;
	cudaEvent_t full_start, full_stop;
	float elapsed_time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&full_start);
	cudaEventCreate(&full_stop);

	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	//Resize and take pointer
	cudaEventRecord(full_start);
	cudaEventSynchronize(full_start);

	cudaEventRecord(start);
	cudaEventSynchronize(start);
	resize(img, img, Size(width, height));
	img.convertTo(img,CV_32F);
	img /= 256.0;
	float* img_ptr = (float*)img.data;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Resize time: " << elapsed_time << endl;

	//Data for FFTW
		
	// Allocate GPU buffers
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	
	cudaMalloc((void**)&in_data, width*height*sizeof(cufftReal));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!");

	cudaMalloc((void**)&out_data, width*height*sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!");

	cudaMalloc((void**)&dev_res, width*height*sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!");

	cudaMalloc((void**)&dev_fft_img, 2*width*height*sizeof(float));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!");

	cudaMalloc((void**)&dev_resComplex, width*height*sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "cudaMalloc failed!");
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Malloc time: " << elapsed_time << endl;
    

	//Copy data to GPU
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	cudaStatus = cudaMemcpy(in_data, img_ptr, width*height*sizeof(cufftReal), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "CPU->GPU cudaMemcpy failed!");
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Memcpy time: " << elapsed_time << endl;
    

	//Create Plans FFT & IFFT
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	if (fft_plan == NULL)
		cufftPlan2d(&fft_plan, width, height, CUFFT_R2C);
	if (ifft_plan == NULL)
		cufftPlan2d(&ifft_plan, width, height, CUFFT_C2R);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Create FFT plans time: " << elapsed_time << endl;

	//Execute Forward FFT
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	cufftExecR2C(fft_plan, in_data, out_data);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	if (TIMER_FLG) cout << "Calculate FFT time: " << elapsed_time << endl;

	//Copy Data to float pointer on GPU    
	cudaStatus = cudaMemcpy(dev_fft_img, out_data, width*height*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) 
				fprintf(stderr, "GPU->CPU cudaMemcpy failed!");
cudaEventRecord(start);
			cudaEventSynchronize(start);
		// Filter Image with Gabor filters in FFT Domain 
		for(int k = 0; k < 32; k++)
	    {	

			multiplyKernel <<< dim3(width/32, height/32), dim3(32, 32)>>> (dev_fft_img, dev_G_data, dev_res, k);
			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

			//Copy data to Complex ppointer on GPU
			cudaStatus = cudaMemcpy(dev_resComplex, dev_res, width*height*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) 
				fprintf(stderr, "GPU->GPU cudaMemcpy failed!");
						
	       //Execute Forward FFT
			cufftExecC2R(ifft_plan, dev_resComplex, in_data);
			cudaDeviceSynchronize();

			//Copy data from GPU	
			cudaStatus = cudaMemcpy(&result[k*width*height], in_data, width*height*sizeof(cufftReal), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess)
				fprintf(stderr, "GPU->CPU cudaMemcpy failed!\n");

			

	    }
		cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed_time, start, stop);
			if (TIMER_FLG) cout << "FFT Convolution time: " << elapsed_time << endl;
		//Clean trash
		cudaEventRecord(start);
		cudaEventSynchronize(start);
		//cufftDestroy(fft_plan);
		//cufftDestroy(ifft_plan);
		cudaFree(in_data);
		cudaFree(out_data);
		//free(buffer);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time, start, stop);
		if (TIMER_FLG) cout << "Clean time: " << elapsed_time << endl;

		cudaEventRecord(full_stop);
		cudaEventSynchronize(full_stop);
		cudaEventElapsedTime(&elapsed_time, full_start, full_stop);
		//cout << "Full calcGist time: " << elapsed_time << endl;

	return result;
}