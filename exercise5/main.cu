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

const float  pi = 3.141592653589793238462;

// uncomment to use the camera
//#define CAMERA
__global__ void convolution(float *d_imgIn, float *d_kernel, float *d_imgOut,
                            int nc, int w, int h, int w_k, int h_k) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  size_t ind = x + (size_t)w * y;

  int mean = w_k / 2;

  if (x < w && y < h) {
    for (int c = 0; c < nc; c++) {
      d_imgOut[ind + w * h * c] = 0;
      for (int k = 0; k < w_k; k++) {
        for (int l = 0; l < h_k; l++) {
          int i_k = x - mean + k;
          int j_k = y - mean + l;

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
  float sigma = 1.0;
  getParam("sigma", sigma, argc, argv);
  cout << "sigma: " << sigma << endl;
// ### Define your own parameters here as needed

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
  //define kernel image
  // cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
  // cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
  // ### Define your own output images here as needed
   cv::Mat mKern(h_k,w_k,CV_32FC1);    // mOut will be a grayscale image, 1 layer

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
	//TODO create function for CPU kernel calc
	float *d_kernel;
	float *d_imgIn;
	float *d_imgOut;
    float *kernel = new float[w_k * w_k]; // height is same as width
	int mean = w_k / 2;
	float sum = 0.0;
    for (int i = 0; i < w_k; i++) {
      for (int j = 0; j < h_k; j++) {

        kernel[j + i * h_k] =
            (1.0f / (2.0f * pi * sigma * sigma))* 
                    exp( -1 * (((i - mean) * (i - mean) + (j - mean) * (j - mean)) /
                               (2 * sigma * sigma)));

		sum += kernel[j + i * h_k];

      }
    }

    // normilize the kernel sum
    float max = 0.0;
    for (int i = 0; i < w_k; i++) {
      for (int j = 0; j < h_k; j++) {

        kernel[j + i * h_k] /= sum;

        if (kernel[j + i * h_k] > max)
          max = kernel[j + i * h_k];
      }
    }

    float *kernel_n = new float[w_k * w_k]; // height is same as width

    for (int i = 0; i < w_k; i++) {
      for (int j = 0; j < h_k; j++) {

        kernel_n[j + i * h_k] = kernel[j + i * h_k] / max;
		
      }
    }

	cudaMalloc(&d_kernel, w_k * h_k * sizeof(float));
    cudaMalloc(&d_imgIn, nc * w * h * sizeof(float));
    cudaMalloc(&d_imgOut, nc * w * h * sizeof(float));

    cudaMemcpy(d_kernel, kernel, w_k * h_k * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgIn, imgIn, nc * w * h * sizeof(float),
               cudaMemcpyHostToDevice);
	
    dim3 block = dim3(32, 8, 1); // 32*8 = 256 threads
    dim3 grid =
        dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);

	convolution <<< grid, block >>> (d_imgIn, d_kernel, d_imgOut, nc, w, h, w_k, h_k);

    cudaMemcpy(imgOut, d_imgOut, nc * w * h * sizeof(float),
                   cudaMemcpyDeviceToHost);

	cudaFree(d_imgIn);
	cudaFree(d_imgOut);
	cudaFree(d_kernel);

    // calculate convolution!
	//TODO init imgOut

    /*for (int c = 0; c < nc; c++)*/
      /*for (int i = 0; i < w; i++) {*/
        /*for (int j = 0; j < h; j++) {*/
          /*imgOut[j * w + i + w * h * c] = 0;*/
          /*for (int k = 0; k < w_k; k++) {*/
            /*for (int l = 0; l < h_k; l++) {*/
              /*int i_k = i - mean + k;*/
              /*int j_k = j - mean + l;*/

              /*if (i_k < 0)*/
                /*i_k = 0;*/
              /*if (i_k > w - 1)*/
                /*i_k = w - 1;*/
              /*if (j_k > h - 1)*/
                /*j_k = h - 1;*/
              /*if (j_k < 0)*/
                /*j_k = 0;*/

              /*imgOut[j * w + i + w * h * c] +=*/
                  /*kernel[l * w_k + k] * imgIn[j_k * w + i_k + w * h * c];*/
            /*}*/
          /*}*/
        /*}*/
      /*}*/

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
    convert_layered_to_mat(mKern, kernel_n);
    showImage("Output", mOut, 100 + w + 40, 100);
    //showImage("Gaussian Kernel", mKern, 100 + w + 40, 100);

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

