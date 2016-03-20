// ###
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

const float pi = 3.141592653589793238462;

__constant__ float c_kernel[41 * 41 * sizeof(float)];

// uncomment to use the camera
//#define CAMERA

void createKernel(float *kernel, float *kernel_n, float sigma, int w, int h) {

  int mean = w / 2;
  float sum = 0.0;

  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {

      // define kernel function on kernel domain
      kernel[j + i * h] =
          (1.0f / (2.0f * pi * sigma * sigma)) *
          exp(-1 * (((i - mean) * (i - mean) + (j - mean) * (j - mean)) /
                    (2 * sigma * sigma)));

      sum += kernel[j + i * h]; // get the integral for normilization purposes
    }
  }

  // normilize the kernel sum
  float max = 0.0;
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {

      kernel[j + i * h] /= sum;

      if (kernel[j + i * h] > max)
        max = kernel[j + i * h];
    }
  }

  for (int i = 0; i < w; i++) {
    for (int j = 0; j < h; j++) {

      kernel_n[j + i * h] = kernel[j + i * h] / max;
    }
  }
}

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

 __device__ void getDiffusion(float *d_v1, float *d_v2, float *d_diffusionTensor, int w, int h,
                                      int nc) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;

  // TODO compute diffusivity factor either here or elsewhere
  float dl_v1 = d_v1[ind];
  float dl_v2 = d_v2[ind];

    d_v1[ind ] = dl_v1 * d_diffusionTensor[ind] + dl_v2 * d_diffusionTensor[ind + 2];
    d_v2[ind ] = dl_v1 * d_diffusionTensor[ind + 1] + dl_v2 * d_diffusionTensor[ind + 3];
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
                          float *d_diffusionTensor, int d_nItterations, float tau, int w,
                          int h, int nc) {

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;
  if( x < w && y < h){
  for (int i = 0; i < d_nItterations; i++) {

    getGradient(d_imgIn, d_imgOutY, d_imgOutX, w, h, nc);
//	 impliment difusion function
	getDiffusion(d_imgOutX, d_imgOutY,d_diffusionTensor, w, h, nc);
	// impliment divergence function
	getDivergence(d_imgOutX, d_imgOutY, d_div, w, h, nc);

	for (int c = 0; c < nc; c++) {

	  d_imgIn[ind + c * w * h] += tau *  d_div[ind + c * w * h];
	}
	}
  }
}

__device__ void getEigen(float *eigenVals, float *eigenVec, float d_t1, float d_t2,
                               float d_t3) {
  // returns eigenvalues of a 2x2 matrix
  float T[4] = {d_t1, d_t2, d_t2, d_t3};
  float trace = T[0] + T[3];
  float determinant = T[0] * T[2] - T[1] * T[3];

  eigenVals[0] = trace / 2.0 + sqrt(trace * trace / (4 - determinant));
  eigenVals[1] = trace / 2.0 - sqrt(trace * trace / (4 - determinant));

  // order the eigen values
  float helper;
  if (eigenVals[0] > eigenVals[1]) {
    helper = eigenVals[0];
    eigenVals[0] = eigenVals[1];
    eigenVals[1] = helper;
  }

  if (T[2] != 0){
	  eigenVec[0] = eigenVals[0] - T[3];
	  eigenVec[1] = eigenVals[1] - T[3];
	  eigenVec[2] = T[2];
	  eigenVec[3] = T[2];
  }else if (T[1] != 0){
	  eigenVec[2] = eigenVals[0] - T[0];
	  eigenVec[3] = eigenVals[1] - T[0];
	  eigenVec[1] = T[1];
	  eigenVec[0] = T[1];
  }
  else if(T[1] == 0 && T[2] == 0){
	  eigenVec[0] = 1;
	  eigenVec[3] = 1;
	  eigenVec[1] = 0;
	  eigenVec[2] = 0;

  }
}


__global__ void setDiffusionTensor(float *d_t1, float *d_t2,
                            float *d_t3, float * d_diffusionTensor, int nc, int w, int h,
                            float alpha, float beta, float C) {

  // setDiffusionTensor() finds features in an image; marking corners red and edges
  // yellow

  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;

// a better solution
  /*struct eigen {*/
	  /*float value;*/
	  /*float *vector;*/

	  /*bool opperator< (&eigen const &other) const {*/
	  /*return value < other.value;*/
	  /*}*/
  /*}*/
  
  float Vals[2] = {0, 0};
  float *eigenVals = Vals;
  float Vec[4];
  float *eigenVec = Vec;
  float u1 = 0;
  float u2 = 0;
  

  getEigen(eigenVals, eigenVec, d_t1[ind], d_t2[ind], d_t2[ind]);
  
  ////This is all for the diffusion tensor
  float other = alpha +
                 (1 - alpha) * exp(-1 * C / ((eigenVals[1] - eigenVals[0]) *
                                             (eigenVals[1] - eigenVals[0])));
  u1 = alpha;

  u2 = (eigenVals[0] == eigenVals[1]) ? alpha : other;

  //Matrix multiplicaiton by hand... to tired for a better implimentation
  d_diffusionTensor[ind + w * h * nc + 0] = u1 * eigenVec[ ind + w * h * nc +  0] * eigenVec[ ind + w * h * nc +  0] + u2 * eigenVec[ ind + w * h * nc +  1] * eigenVec[ ind + w * h * nc +  1];
  d_diffusionTensor[ind + w * h * nc + 1] = u1 * eigenVec[ ind + w * h * nc +  0] * eigenVec[ ind + w * h * nc +  2] + u2 * eigenVec[ ind + w * h * nc +  1] * eigenVec[ ind + w * h * nc +  3];
  d_diffusionTensor[ind + w * h * nc + 2] = u1 * eigenVec[ ind + w * h * nc +  2] * eigenVec[ ind + w * h * nc +  0] + u2 * eigenVec[ ind + w * h * nc +  1] * eigenVec[ ind + w * h * nc +  3];
  d_diffusionTensor[ind + w * h * nc + 3] = u1 * eigenVec[ ind + w * h * nc +  2] * eigenVec[ ind + w * h * nc +  2] + u2 * eigenVec[ ind + w * h * nc +  3] * eigenVec[ ind + w * h * nc +  3];

  /////////
  
}

__global__ void globalConvolution(float *d_imgIn, float *d_kernel,
                                  float *d_imgOut, int nc, int w, int h,
                                  int w_k, int h_k) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;

  int mid = w_k / 2;

  if (x < w && y < h) {
    // itterate over colors
    for (int c = 0; c < nc; c++) {
      // initialize output image to 0
      d_imgOut[ind + w * h * c] = 0;
      // itterate over kernel funtion
      for (int k = 0; k < w_k; k++) {
        for (int l = 0; l < h_k; l++) {
          int i_k = x - mid + k;
          int j_k = y - mid + l;

          // check boundary conditions
          if (i_k < 0)
            i_k = 0;
          if (i_k > w - 1)
            i_k = w - 1;
          if (j_k > h - 1)
            j_k = h - 1;
          if (j_k < 0)
            j_k = 0;

          d_imgOut[ind + w * h * c] +=
              d_kernel[l * w_k + k] * d_imgIn[j_k * w + i_k + w * h * c];
        }
      }
    }
  }
}

__global__ void computeM(float *d_gradStencilX, float *d_gradStencilY,
                         float *d_m1, float *d_m2, float *d_m3, int w, int h,
                         int nc) {
  // image coordinates
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;

  if (x < w && y < h) {
    d_m1[ind] = 0;
    d_m2[ind] = 0;
    d_m3[ind] = 0;

    for (int i = 0; i < nc; i++) {
      d_m1[ind] += d_gradStencilX[ind + w * h * i] *
                   d_gradStencilX[ind + w * h * i] * 10;
      d_m2[ind] += d_gradStencilX[ind + w * h * i] *
                   d_gradStencilY[ind + w * h * i] * 10;
      d_m3[ind] += d_gradStencilY[ind + w * h * i] *
                   d_gradStencilY[ind + w * h * i] * 10;
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
  float sigma = 0.5;
  getParam("sigma", sigma, argc, argv);
  cout << "sigma: " << sigma << endl;
  // ### Define your own parameters here as needed

  float alpha = 0.01;
  getParam("alpha", alpha, argc, argv);
  cout << "alpha: " << alpha << endl;

  float beta = 0.001;
  getParam("beta", beta, argc, argv);
  cout << "beta: " << beta << endl;

  float gain = 1.0;
  getParam("gain", gain, argc, argv);
  cout << "gain: " << gain << endl;

  float diffusion = 1;
  getParam("diffusion", diffusion, argc, argv);
  cout << "diffusion: " << diffusion << endl;

  float tau = 0.02;
  getParam("tau", tau, argc, argv);
  cout << "tau: " << tau << endl;

  int N = 1;
  getParam("N", N, argc, argv);
  cout << "N: " << N << endl;

  float C = 0.00005;
  getParam("C", C, argc, argv);
  cout << "C: " << C << endl;

  int roh = 3;
  getParam("roh", roh, argc, argv);
  cout << "roh: " << roh << endl;
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
  int r = ceil(3 * sigma);
  int w_k = r * 2 + 1;
  int h_k = w_k;
  cout << "image: " << w << " x " << h << endl;

  // Set the output image format
  // ###
  // ###
  // ### TODO: Change the output image format as needed
  // ###
  // ###
  cv::Mat mOut(h, w, mIn.type()); // mOut will have the same number of channels
                                  // as the input image, nc layers
  // define kernel image
  // cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
  // cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
  // ### Define your own output images here as needed
  cv::Mat mKern(h_k, w_k, CV_32FC1); // mOut will be a grayscale image, 1 layer
  cv::Mat mM1(h, w, CV_32FC1);       // mOut will be a grayscale image, 1 layer
  cv::Mat mM2(h, w, CV_32FC1);       // mOut will be a grayscale image, 1 layer
  cv::Mat mM3(h, w, CV_32FC1);       // mOut will be a grayscale image, 1 layer

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
    float *kernel = new float[w_k * w_k];   // height is same as width
    float *kernel_n = new float[w_k * w_k]; // height is same as width
    float *m1 = new float[w * h];
    float *m2 = new float[w * h];
    float *m3 = new float[w * h];

    float stencilX[9] = {-3, 0, 3, -10, 0, 10, -3, 0, 3};
    float *gradStencilX = stencilX;
    float stencilY[9] = {-3, -10, -3, 0, 0, 0, 3, 10, 3};
    float *gradStencilY = stencilY;
    // normalize values
    for (int i = 0; i < 9; i++) {
      // cout<< "X_0: " << stencilX[i] << "; ";
      stencilX[i] /= 32.0;
      stencilY[i] /= 32.0;
      // cout<< "X_1: " << stencilX[i] << endl;
    }

	float *d_imgOutX;
	float *d_imgOutY;
    float *d_gradStencilX = new float[3 * 3];
    float *d_gradStencilY = new float[3 * 3];
    float *d_gradX;
    float *d_gradY;
    float *d_m1;
    float *d_m2;
    float *d_m3;
    float *d_pm1;
    float *d_pm2;
    float *d_pm3;
    float *d_t1;
    float *d_t2;
    float *d_t3;
    float *d_kernel;
    float *d_imgIn;
    float *d_imgOut;
	float *d_div;
	float *d_diffusionTensor;

    createKernel(kernel, kernel_n, sigma, w_k, h_k);

	cudaMalloc(&d_diffusionTensor, 4 * nc * w * h * sizeof(float));
	CUDA_CHECK;
	cudaMalloc(&d_div, nc * w * h * sizeof(float));
	CUDA_CHECK;
    cudaMalloc(&d_kernel, w_k * h_k * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_imgIn, nc * w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_imgOut, nc * w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_gradStencilX, 3 * 3 * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_gradStencilY, 3 * 3 * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_gradX, nc * w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_gradY, nc * w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_m1, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_m2, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_m3, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_pm1, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_pm2, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_pm3, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_t1, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_t2, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_t3, w * h * sizeof(float));
    CUDA_CHECK;
    cudaMalloc(&d_imgIn, nc * w * h * sizeof(float));
	CUDA_CHECK;
    cudaMalloc(&d_imgOutY, nc * w * h * sizeof(float));
	CUDA_CHECK;
    cudaMalloc(&d_imgOutX, nc * w * h * sizeof(float));
	CUDA_CHECK;


    cudaMemcpy(d_kernel, kernel, w_k * h_k * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, nc * w * h * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK;

    cudaMemcpy(d_gradStencilX, gradStencilX, 3 * 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK;
    cudaMemcpy(d_gradStencilY, gradStencilY, 3 * 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK;

    dim3 block = dim3(32, 8, 1); // 32*8 = 256 threads
    dim3 grid =
        dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
    CUDA_CHECK;

    // calculate convolution!

    globalConvolution<<<grid, block>>>(d_imgIn, d_kernel, d_imgOut, nc, w, h,
                                       h_k, w_k);
    // calculate gradient of convoluted image

    // for X
    globalConvolution<<<grid, block>>>(d_imgOut, d_gradStencilX, d_gradX, nc, w,
                                       h, 3, 3);
    // for Y
    globalConvolution<<<grid, block>>>(d_imgOut, d_gradStencilY, d_gradY, nc, w,
                                       h, 3, 3);

    computeM<<<grid, block>>>(d_gradX, d_gradY, d_m1, d_m2, d_m3, w, h, nc);

    // get structure tensor values
    globalConvolution<<<grid, block>>>(d_m1, d_kernel, d_t1, 1, w, h, w_k, h_k);

    globalConvolution<<<grid, block>>>(d_m2, d_kernel, d_t2, 1, w, h, w_k, h_k);

    globalConvolution<<<grid, block>>>(d_m3, d_kernel, d_t3, 1, w, h, w_k, h_k);

    // post smooth structure tensor values

    createKernel(kernel, kernel_n, roh, w_k, h_k);

    cudaMemcpy(d_kernel, kernel, w_k * h_k * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK;

	//post smothing
	globalConvolution<<<grid, block>>>(d_t1, d_kernel, d_pm1, 1, w, h, w_k,
									   h_k);

	globalConvolution<<<grid, block>>>(d_t2, d_kernel, d_pm2, 1, w, h, w_k,
									   h_k);

	globalConvolution<<<grid, block>>>(d_t3, d_kernel, d_pm3, 1, w, h, w_k,
			h_k);

	//TODO allocate diffusion tensor
	setDiffusionTensor<<<grid, block>>>(d_pm1, d_pm2, d_pm3, d_diffusionTensor, nc, w, h, alpha, beta, C);



	/*updateImg<<<grid, block>>>(d_imgIn, d_pm1, d_imgOutX, d_imgOutY, d_diffusionTensor, N, tau, w,*/
							   /*h, nc);*/



	cudaMemcpy(imgOut, d_imgOut, nc * w * h * sizeof(float),
	cudaMemcpyDeviceToHost);
    /*cudaMemcpy(imgOut, d_gradY, nc * w * h * sizeof(float),*/
               /*cudaMemcpyDeviceToHost);*/
    cudaMemcpy(m1, d_t1, w * h * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m2, d_t2, w * h * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m3, d_t3, w * h * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_imgIn);
    cudaFree(d_imgOut);
    cudaFree(d_kernel);
	cudaFree(d_div);
    // TODO free memory for stuff

    timer.end();
    float t = timer.get(); // elapsed time in seconds
    cout << "time: " << t * 1000 << " ms" << endl;

    // show input image
    showImage("Input", mIn, 100,
              100); // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the
    // layered raw array
    convert_layered_to_mat(mOut, imgIn);
    convert_layered_to_mat(mKern, kernel_n);
    convert_layered_to_mat(mM1, m1);
    convert_layered_to_mat(mM2, m2);
    convert_layered_to_mat(mM3, m3);

	 //showImage("Output", mOut * gain , 100 + w + 40, 100);
	showImage("m1", mM1 * 10, 400, 100);
/*showImage("m2", mM2 * 10, 50 + w, 100);*/
/*showImage("m3", mM3 * 10, 50 + 2 * w, 100);*/
// showImage("Gaussian Kernel", mKern, 100 + w + 40, 100);

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

