
#ifndef SCM_DATA_VOLUME_FILTER_KERNEL_SOBEL3X3_H_INCLUDED
#define SCM_DATA_VOLUME_FILTER_KERNEL_SOBEL3X3_H_INCLUDED

#include <scm/data/volume/volume_filter_kernel.h>

namespace scm {
namespace data {


// kernel layout in x-major order
// 
// 0 is origin
// xy slices
// [ 6  7  8] [15 16 17] [24 25 26]
// [ 3  4  5] [12 13 14] [21 22 23]
// [ 0  1  2] [ 9 10 11] [18 19 20]

struct volume_filter_kernel_sobel3x3_x : public volume_filter_kernel
{
    volume_filter_kernel_sobel3x3_x() : volume_filter_kernel() {

        _width = _height = _depth = 3;
        _weight = 48.0f;

        _values.push_back(-1.f);  _values.push_back(0.f);  _values.push_back(1.f);
        _values.push_back(-3.f);  _values.push_back(0.f);  _values.push_back(3.f);
        _values.push_back(-1.f);  _values.push_back(0.f);  _values.push_back(1.f);

        _values.push_back(-3.f);  _values.push_back(0.f);  _values.push_back(3.f);
        _values.push_back(-6.f);  _values.push_back(0.f);  _values.push_back(6.f);
        _values.push_back(-3.f);  _values.push_back(0.f);  _values.push_back(3.f);

        _values.push_back(-1.f);  _values.push_back(0.f);  _values.push_back(1.f);
        _values.push_back(-3.f);  _values.push_back(0.f);  _values.push_back(3.f);
        _values.push_back(-1.f);  _values.push_back(0.f);  _values.push_back(1.f);
    }
};

struct volume_filter_kernel_sobel3x3_y : public volume_filter_kernel
{
    volume_filter_kernel_sobel3x3_y() : volume_filter_kernel() {

        _width = _height = _depth = 3;
        _weight = 48.0f;

        _values.push_back(-1.f);  _values.push_back(-3.f);  _values.push_back(-1.f);
        _values.push_back( 0.f);  _values.push_back( 0.f);  _values.push_back( 0.f);
        _values.push_back( 1.f);  _values.push_back( 3.f);  _values.push_back( 1.f);

        _values.push_back(-3.f);  _values.push_back(-6.f);  _values.push_back(-3.f);
        _values.push_back( 0.f);  _values.push_back( 0.f);  _values.push_back( 0.f);
        _values.push_back( 3.f);  _values.push_back( 6.f);  _values.push_back( 3.f);

        _values.push_back(-1.f);  _values.push_back(-3.f);  _values.push_back(-1.f);
        _values.push_back( 0.f);  _values.push_back( 0.f);  _values.push_back( 0.f);
        _values.push_back( 1.f);  _values.push_back( 3.f);  _values.push_back( 1.f);
    }
};

struct volume_filter_kernel_sobel3x3_z : public volume_filter_kernel
{
    volume_filter_kernel_sobel3x3_z() : volume_filter_kernel() {

        _width = _height = _depth = 3;
        _weight = 48.0f;

        _values.push_back(-1.f);  _values.push_back(-3.f);  _values.push_back(-1.f);
        _values.push_back(-3.f);  _values.push_back(-6.f);  _values.push_back(-3.f);
        _values.push_back(-1.f);  _values.push_back(-3.f);  _values.push_back(-1.f);

        _values.push_back( 0.f);  _values.push_back( 0.f);  _values.push_back( 0.f);
        _values.push_back( 0.f);  _values.push_back( 0.f);  _values.push_back( 0.f);
        _values.push_back( 0.f);  _values.push_back( 0.f);  _values.push_back( 0.f);

        _values.push_back( 1.f);  _values.push_back( 3.f);  _values.push_back( 1.f);
        _values.push_back( 3.f);  _values.push_back( 6.f);  _values.push_back( 3.f);
        _values.push_back( 1.f);  _values.push_back( 3.f);  _values.push_back( 1.f);
    }
};

} // namespace scm
} // namespace data

#endif // SCM_DATA_VOLUME_FILTER_KERNEL_SOBEL3X3_H_INCLUDED
