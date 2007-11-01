
#ifndef SCM_DATA_HISTOGRAM_1D_CALCULATOR_H_INCLUDED
#define SCM_DATA_HISTOGRAM_1D_CALCULATOR_H_INCLUDED

#include <scm/data/analysis/regular_grid_data_3d.h>
#include <scm/data/analysis/histogram/histogram_1d.h>

namespace scm {
namespace data {

template<typename val_type>
class histogram_1d_calculator
{
    // do not instantiiate!
    BOOST_STATIC_ASSERT(sizeof(val_type) == 0);
}; // class histogram_calculator

template<>
class histogram_1d_calculator<unsigned char>
{
public:
    static bool calculate(scm::histogram_1d<unsigned char>& histogram,
                          const scm::regular_grid_data_3d<unsigned char>& data,
                          unsigned num_bins = 0);

}; // class histogram_calculator

} // namespace data
} // namespace scm

#include "histogram_1d_calculator.inl"

#endif // SCM_DATA_HISTOGRAM_1D_CALCULATOR_H_INCLUDED

