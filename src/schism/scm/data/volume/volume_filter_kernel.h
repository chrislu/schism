
#ifndef SCM_DATA_VOLUME_FILTER_KERNEL_H_INCLUDED
#define SCM_DATA_VOLUME_FILTER_KERNEL_H_INCLUDED

#include <vector>
#include <algorithm>

namespace scm {
namespace data {

// base class for a 3d filter kernel
// layout as folows:
// _values holds w*h*d values storing in x - major order
// starting at the origin
//
struct volume_filter_kernel
{
    volume_filter_kernel() {
        _width = _height = _depth = 0;
        _weight = 0.0f;
    }
    virtual ~volume_filter_kernel() {
    }

    float               apply(const std::vector<float>& inp) {
        float sum = 0.f;

        if (_values.size() == inp.size()) {

            for (std::vector<float>::size_type i = 0; i < _values.size(); i++) {
                sum += inp[i] * _values[i];
            }

            sum /= _weight;
        }
        return (sum);
    }

    unsigned int        _width;
    unsigned int        _height;
    unsigned int        _depth;

    float               _weight;

    std::vector<float>  _values;

}; // struct volume_filter_kernel

} // namespace data
} // namespace scm

#endif // SCM_DATA_VOLUME_FILTER_KERNEL_H_INCLUDED
