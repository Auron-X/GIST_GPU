#include <opencv2\opencv.hpp>
#include <iostream>
#include <ctime>
#include "fftw3.h"
//#include "cufftw.h"

using namespace std;
using namespace cv;

# define PI           3.14159265358979323846
# define N			  256

const int numOfScales = 4;
const int numOfOrient = 8;
const int numOfFilters = numOfScales*numOfOrient;
const int filterSizeX = N;
const int filterSizeY = N;	
const int width = N;
const int height = N;	


void fftshift(float *data, int w, int h)
{
    int i, j;

    float *buff = (float *) malloc(w*h*sizeof(float));

    memcpy(buff, data, w*h*sizeof(float));

    for(j = 0; j < (h+1)/2; j++)
    {
        for(i = 0; i < (w+1)/2; i++) {
            data[(j+h/2)*w + i+w/2] = buff[j*w + i];
        }

        for(i = 0; i < w/2; i++) {
            data[(j+h/2)*w + i] = buff[j*w + i+(w+1)/2];
        }
    }

    for(j = 0; j < h/2; j++)
    {
        for(i = 0; i < (w+1)/2; i++) {
            data[j*w + i+w/2] = buff[(j+(h+1)/2)*w + i];
        }

        for(i = 0; i < w/2; i++) {
            data[j*w + i] = buff[(j+(h+1)/2)*w + i+(w+1)/2];
        }
    }

    free(buff);
}

extern float* generateGaborFiltersGPU();
extern float* calcGistGPU(Mat img, float* G);


vector <Mat> generateGaborFiltersCPU()
{
	vector <Mat> G;

	
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
	
	
	
	
	//Meshgrid
	float *fx = (float *) malloc(filterSizeX*filterSizeY*sizeof(float));
    float *fy = (float *) malloc(filterSizeX*filterSizeY*sizeof(float));
    float *fr = (float *) malloc(filterSizeX*filterSizeY*sizeof(float));
    float *t  = (float *) malloc(filterSizeX*filterSizeY*sizeof(float));
	float *tr  = (float *) malloc(filterSizeX*filterSizeY*sizeof(float));


	
	for (int j=0; j<filterSizeY; j++)
		for (int i=0; i<filterSizeX; i++)
		{
			fx[j*filterSizeX + i] = (float) i - filterSizeX/2.0f;
            fy[j*filterSizeX + i] = (float) j - filterSizeY/2.0f;
            fr[j*filterSizeX + i] = sqrt(fx[j*filterSizeX + i]*fx[j*filterSizeX + i] + fy[j*filterSizeX + i]*fy[j*filterSizeX + i]);
            t[j*filterSizeX + i]  = atan2(fy[j*filterSizeX + i], fx[j*filterSizeX + i]);
		}
	fftshift(fr, filterSizeX, filterSizeY);
    fftshift(t, filterSizeX, filterSizeY);
	
	//Gabor Coefficients calculation
	for(int fn = 0; fn < numOfFilters; fn++)
    {
        Mat G0(filterSizeX, filterSizeY, CV_32F);
		float* G_data = (float*)G0.data;
        float *f_ptr = t;
        float *fr_ptr = fr;

        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++)
            {
				float tmp = *f_ptr++ + gaborParams[fn][3];

                if(tmp < -PI) {
                    tmp += 2.0f*PI;
                }
                else if (tmp > PI) {
                    tmp -= 2.0f*PI;
                }
				
                G_data[j*width+i] = exp(-10.0f*gaborParams[fn][0]*(*fr_ptr/height/gaborParams[fn][1]-1)*
					(*fr_ptr/width/gaborParams[fn][1]-1)-2.0f*gaborParams[fn][2]*PI*tmp*tmp);
                fr_ptr++;
				
            }
        }

       G.push_back(G0);
    }
	
	
	//Print Gabor Filter values
	/*float* G_data = (float*)G[0].data;
	for (int j=0; j<10; j++)
	{
		for (int i=0; i<10; i++)
		{ 
			cout << G_data[j*filterSizeX+i] << " ";
		}
		cout << endl;
	}*/

	
	
	
    free(fx);
    free(fy);
    free(fr);
    free(t);
	
	return G;
}

float* calcGistCPU(Mat img, float* G)
{
	//Resize and take pointer
	resize(img, img, Size(width, height));
	unsigned char* img_ptr = (unsigned char*)img.data;

	//Result - Filtered Image
	float *result = (float *) malloc(width*height*numOfFilters*sizeof(float));

	//Data for FFTW
    fftwf_complex *orig_img_cmp = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    fftwf_complex *filt_img_cmp = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    fftwf_complex *fft_image_cmp = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));
    fftwf_complex *filt_fft_img_cmp = (fftwf_complex *) fftwf_malloc(width*height*sizeof(fftwf_complex));


	for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
			//Scale down to 0.0 ~ 1.0
            orig_img_cmp[j*width + i][0] = img_ptr[j*width+i]/256.0;
            orig_img_cmp[j*width + i][1] = 0.0f;
        }
    }

	// Create FFT and IFFT Plans
    fftwf_plan fft = fftwf_plan_dft_2d(width, height, orig_img_cmp, fft_image_cmp, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan ifft = fftwf_plan_dft_2d(width, height, filt_fft_img_cmp, filt_img_cmp, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Convert Image to FFT Domain
	fftwf_execute(fft);

	// Filter Image with Gabor filters in FFT Domain 
	for(int k = 0; k < 32; k++)
    {
        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++)
            {
                filt_fft_img_cmp[j*width+i][0] = fft_image_cmp[j*width+i][0] * G[k*width*height+j*width+i];
                filt_fft_img_cmp[j*width+i][1] = fft_image_cmp[j*width+i][1] * G[k*width*height+j*width+i];
            }
        }

        fftwf_execute(ifft);

        for(int j = 0; j < height; j++)
        {
            for(int i = 0; i < width; i++) {
                result[k*width*height+j*width+i] = sqrt(filt_img_cmp[j*width+i][0]*filt_img_cmp[j*width+i][0]+filt_img_cmp[j*width+i][1]*filt_img_cmp[j*width+i][1]);
            }
        }
    }
	
	//Clean FFTW trash
	fftwf_destroy_plan(fft);
    fftwf_destroy_plan(ifft);

    fftwf_free(orig_img_cmp);
	fftwf_free(filt_img_cmp);
    fftwf_free(fft_image_cmp);
    fftwf_free(filt_fft_img_cmp);

	return result;
}

void testGenerateGabor(int TEST_LOOPS = 1)
{
	float start_CPU = getTickCount();
	for (int z=0; z<TEST_LOOPS; z++)
		vector <Mat> G = generateGaborFiltersCPU();

	float end_CPU = getTickCount();
	float CPU_elpTime = (end_CPU-start_CPU)/getTickFrequency();
	cout << "CPU Time:   " << CPU_elpTime*1000/TEST_LOOPS << endl;

	float* G;
	float start_GPU = getTickCount();
	for (int z=0; z<TEST_LOOPS; z++)
		G = generateGaborFiltersGPU();

	////Print Gabor Filter values
	//for (int j=0; j<10; j++)
	//{
	//	for (int i=0; i<10; i++)
	//	{ 
	//		cout << G[j*256+i] << " ";
	//	}
	//	cout << endl;
	//}
	

	float end_GPU = getTickCount();
	float GPU_elpTime = (end_GPU-start_GPU)/getTickFrequency();

	cout << "GPU Time:   " << GPU_elpTime*1000/TEST_LOOPS << endl;
	cout << endl;

	cout << "Ratio:   " << CPU_elpTime/GPU_elpTime << endl;
}

void testCalcGist(Mat img, float* G, int TEST_LOOPS = 1)
{
	float start_CPU = getTickCount();
	float* res;
	for (int z=0; z<TEST_LOOPS; z++)
		res = calcGistCPU(img,G);

	float end_CPU = getTickCount();
	float CPU_elpTime = (end_CPU-start_CPU)/getTickFrequency();
	cout << "gistCalc - CPU Time:   " << CPU_elpTime*1000/TEST_LOOPS << endl;

	
	//Print Result
	/*for (int j=0; j<5; j++)
	{
		for (int i=0; i<5; i++)
		{ 
			cout << res[j*filterSizeX+i] << " ";
		}
		cout << endl;
	}*/

	float start_GPU = getTickCount();
	for (int z=0; z<TEST_LOOPS; z++)
	
		res = calcGistGPU(img, G);
	
	float end_GPU = getTickCount();
	float GPU_elpTime = (end_GPU-start_GPU)/getTickFrequency();

	//Print Result
	/*for (int j=0; j<5; j++)
	{
		for (int i=0; i<5; i++)
		{ 
			cout << res[j*filterSizeX+i] << " ";
		}
		cout << endl;
	}*/

	cout << "GPU Time:   " << GPU_elpTime*1000/TEST_LOOPS << endl;
	cout << endl;

	cout << "Ratio:   " << CPU_elpTime/GPU_elpTime << endl;
}

int main()
{
	//Read Image in Grayscale
	Mat img = imread("1.jpg", 0);
	//testGenerateGabor(100);


	float* G;
	G = generateGaborFiltersGPU();
	
	float* res;
	
	testGenerateGabor(5);
	//testCalcGist(img, G, 5);
	
	getchar();
	return 0;
}