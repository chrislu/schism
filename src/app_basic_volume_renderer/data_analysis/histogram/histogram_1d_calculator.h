
#ifndef HISTOGRAM_1D_CALCULATOR_H_INCLUDED
#define HISTOGRAM_1D_CALCULATOR_H_INCLUDED

#include <data_analysis/regular_grid_data_3d.h>
#include <data_analysis/histogram/histogram_1d.h>

namespace scm
{
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

} // namespace scm

#include "histogram_1d_calculator.inl"

#endif // HISTOGRAM_1D_CALCULATOR_H_INCLUDED



