
#ifndef GAUSSIAN_BLUR_H_INCLUDED
#define GAUSSIAN_BLUR_H_INCLUDED

extern "C"
{
    void gaussian_blur_7x7(int* in_data, int* out_data, unsigned width, unsigned height);
    extern float gauss_kernel_7x7[49];
}


#endif // GAUSSIAN_BLUR_H_INCLUDED
