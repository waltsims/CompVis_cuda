// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Winter Semester 2015/2016, March 15 - April 15
// ###
// ###

#include "helper.h"
#include <iostream>
using namespace std;

// uncomment to use the camera
//#define CAMERA

__device__ void forwardDifferenceX(float *du, float *u, int ind) {
  du[ind] = u[ind + 1] - u[ind];
}

__device__ void forwardDifferenceY(float *du, float *u, int ind, int w) {
  du[ind] = u[ind + w] - u[ind];
}

__device__ float backwardsDifferenceX(float *u, int ind) {
  return u[ind] - u[ind - 1];
}

__device__ float backwardsDifferenceY(float *u, int ind, int w) {
  return u[ind ] - u[ind - w];
}

 __device__ void getDiffusion(float *d_v1, float *d_v2, int w, int h,
                                      int nc) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;

  // TODO compute diffusivity factor either here or elsewhere
  float eps = 0.01;
  float factorX = 1;
  float factorY = 1;
  /*float factorX = 1.f/max (d_v1[ind], eps);*/
  /*float factorY = 1.f/max (d_v2[ind], eps);*/
  /*float factorX = exp( (d_v1[ind] * d_v1[ind]) * -1 / eps) / eps ;*/
  /*float factorY = exp( (d_v2[ind] * d_v2[ind]) * -1 / eps) / eps ;*/

    d_v1[ind ] *= factorX;
    d_v2[ind ] *= factorY;
}


__device__ void getGradient(float * d_imgIn, float *d_imgOutY, float *d_imgOutX, int w,
                         int h, int nc) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;

  if (x < w && y < h) {
    for (int c = 0; c < nc; c++) {
		d_imgOutY[ind + w * h * c] = 0;
		d_imgOutX[ind + w * h * c] = 0;
	  if (y + 1 < h) {
		forwardDifferenceY(d_imgOutY,  d_imgIn, c * w * h + ind, w);
	  } else
		d_imgOutY[c * w * h + ind] = 0;
	  if (x + 1 < w) {
		forwardDifferenceX(d_imgOutX,  d_imgIn, c * w * h + ind);
	  } else
		d_imgOutX[c * w * h + ind] = 0;
	}
  }
}

__device__ void getDivergence(float *v1, float *v2, float *d_div, int w, int h,
                           int nc) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;
  float dv1; 
  float dv2;
  if (x < w && y < h) {
    for (int c = 0; c < nc; c++) {
      size_t ind_loc = c * w * h + ind;
      if (y  > 0) {
        dv1 = backwardsDifferenceX( v1, ind_loc);
      } else
        dv1= v1[ind_loc];
      if (x  > 0) {
        dv2 = backwardsDifferenceY( v2, ind_loc, w);
      } else
        dv2= v2[ind_loc];
      d_div[ind_loc] = dv1 + dv2;
    }
  }
}


__global__ void updateImg(float *d_imgIn, float * d_div, float *d_imgOutX, float *d_imgOutY,
                          int d_nItterations, float tau, int w,
                          int h, int nc) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;
  if( x < w && y < h){
  for (int i = 0; i < d_nItterations; i++) {

    getGradient(d_imgIn, d_imgOutY, d_imgOutX, w, h, nc);
//	 impliment difusion function
	getDiffusion(d_imgOutX, d_imgOutY, w, h, nc);
	// impliment divergence function
	getDivergence(d_imgOutX, d_imgOutY, d_div, w, h, nc);

	for (int c = 0; c < nc; c++) {

	  d_imgIn[ind + c * w * h] += tau *  d_div[ind + c * w * h];
	}
	}
  }
}

int main(int argc, char **argv) {
  // Before the GPU can process your kernels, a so called "CUDA context" must be
  // initialized
  // This happens on the very first call to a CUDA function, and takes some time
  // (around half a second)
  // We will do it right here, so that the run time measurements are accurate
  cudaDeviceSynchronize();
  CUDA_CHECK;

// Reading command line parameters:
// getParam("param", var, argc, argv) looks whether "-param xyz" is specified,
// and if so stores the value "xyz" in "var"
// If "-param" is not specified, the value of "var" remains unchanged
//
// return value: getParam("param", ...) returns true if "-param" is specified,
// and false otherwise

#ifdef CAMERA
#else
  // input image
  string image = "";
  bool ret = getParam("i", image, argc, argv);
  if (!ret)
    cerr << "ERROR: no image specified" << endl;
  if (argc <= 1) {
    cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]"
         << endl;
    return 1;
  }
#endif

  // number of computation repetitions to get a better run time measurement
  int repeats = 1;
  getParam("repeats", repeats, argc, argv);
  cout << "repeats: " << repeats << endl;

  // load the input image as grayscale if "-gray" is specifed
  bool gray = false;
  getParam("gray", gray, argc, argv);
  cout << "gray: " << gray << endl;

// ### Define your own parameters here as needed

  float diffusion = 1;
  getParam("diffusion", diffusion, argc, argv);
  cout << "diffusion: " << diffusion << endl;

  float tau = 0.02;
  getParam("tau", tau, argc, argv);
  cout << "tau: " << tau << endl;

  int N = 1;
  getParam("N", N, argc, argv);
  cout << "N: " << N << endl;
// Init camera / Load input image
#ifdef CAMERA

  // Init camera
  cv::VideoCapture camera(0);
  if (!camera.isOpened()) {
    cerr << "ERROR: Could not open camera" << endl;
    return 1;
  }
  int camW = 640;
  int camH = 480;
  camera.set(CV_CAP_PROP_FRAME_WIDTH, camW);
  camera.set(CV_CAP_PROP_FRAME_HEIGHT, camH);
  // read in first frame to get the dimensions
  cv::Mat mIn;
  camera >> mIn;

#else

  // Load the input image using opencv (load as grayscale if "gray==true",
  // otherwise as is (may be color or grayscale))
  cv::Mat mIn =
      cv::imread(image.c_str(), (gray ? CV_LOAD_IMAGE_GRAYSCALE : -1));
  // check
  if (mIn.data == NULL) {
    cerr << "ERROR: Could not load image " << image << endl;
    return 1;
  }

#endif

  // convert to float representation (opencv loads image values as single bytes
  // by default)
  mIn.convertTo(mIn, CV_32F);
  // convert range of each channel to [0,1] (opencv default is [0,255])
  mIn /= 255.f;
  // get image dimensions
  int w = mIn.cols;        // width
  int h = mIn.rows;        // height
  int nc = mIn.channels(); // number of channels
  cout << "image: " << w << " x " << h << endl;

  // Set the output image format
  // ###
  // ###
  // ### TODO: Change the output image format as needed
  // ###
  // ###
 // cv::Mat mOut(h, w, mIn.type()); // mOut will have the same number of channels
                                  // as the input image, nc layers
   cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
   //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
  // ### Define your own output images here as needed

  // Allocate arrays
  // input/output image width: w
  // input/output image height: h
  // input image number of channels: nc
  // output image number of channels: mOut.channels(), as defined above (nc, 3,
  // or 1)

  // allocate raw input image array
  float *imgIn = new float[(size_t)w * h * nc];

  // allocate raw output array (the computation result will be stored in this
  // array, then later converted to mOut for displaying)
  float *imgOut = new float[(size_t)w * h * mOut.channels()];

// For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
  // Read a camera image frame every 30 milliseconds:
  // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
  // returns a value <0 if no key is pressed during this time, returns
  // immediately with a value >=0 if a key is pressed
  while (cv::waitKey(30) < 0) {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single
    // bytes by default)
    mIn.convertTo(mIn, CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg...
    // bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for
    // loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered(imgIn, mIn);

    Timer timer;
    timer.start();
    // ###
    // ###

cout << "this is real"	<< endl;

    float *d_imgIn;
    float *d_imgOutX;
    float *d_imgOutY;
	float *d_lap;
	float *lap = new float[w * h];
	float *d_div;
		
    cudaMalloc(&d_imgIn, nc * w * h * sizeof(float));
    cudaMalloc(&d_imgOutY, nc * w * h * sizeof(float));
    cudaMalloc(&d_imgOutX, nc * w * h * sizeof(float));
	cudaMalloc(&d_div, nc * w * h * sizeof(float));
    cudaMalloc(&d_lap, w * h * sizeof(float));

    cudaMemcpy(d_imgIn, imgIn, nc * w * h * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 block = dim3(32, 8, 1); // 32*8 = 256 threads
    dim3 grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

    //  impliment l2 norm function
    updateImg<<<grid, block>>>(d_imgIn, d_div, d_imgOutX, d_imgOutY, N, tau, w,
                               h, nc);

    //l_2<<<grid, block>>>(d_div, d_lap, w, h, nc);


	
    cudaMemcpy(imgOut, d_imgIn, nc * w * h * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_imgOutY);
	cudaFree(d_imgOutX);
    cudaFree(d_div);
	cudaFree(d_imgIn);
	cudaFree(d_lap);

    // ###
    // ###
    timer.end();
    float t = timer.get(); // elapsed time in seconds
    cout << "time: " << t * 1000 << " ms" << endl;

    // show input image
    showImage("Input", mIn, 100,
              100); // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the
    // layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut , 100 + w + 40, 100);

// ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
  }
#else
  // wait for key inputs
  cv::waitKey(0);
#endif

  // save input and result
  cv::imwrite("image_input.png",
              mIn * 255.f); // "imwrite" assumes channel range [0,255]
  cv::imwrite("image_result.png", mOut * 255.f);

  // free allocated arrays
  delete[] imgIn;
  delete[] imgOut;

  // close all opencv windows
  cvDestroyAllWindows();
  return 0;
}

