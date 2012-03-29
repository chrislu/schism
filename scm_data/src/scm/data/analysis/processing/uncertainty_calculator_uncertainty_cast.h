
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_DATA_UNCERTAINTY_CALCULATOR_UNCERTAINTY_CAST_H_INCLUDED
#define SCM_DATA_UNCERTAINTY_CALCULATOR_UNCERTAINTY_CAST_H_INCLUDED

#include <scm/data/analysis/regular_grid_data_3d.h>
#include <scm/data/analysis/regular_grid_data_3d_write_accessor.h>
#include <scm/data/analysis/transfer_function/piecewise_function_1d.h>

namespace scm {
namespace data {

template<typename value_type>
class uncertainty_calculator_uncertainty_cast : public scm::regular_grid_data_3d_write_accessor<value_type>
{
public:
    uncertainty_calculator_uncertainty_cast(){
    }

    bool            calculate_uncertainty_data(const scm::regular_grid_data_3d<value_type>& source_data,
                                               scm::regular_grid_data_3d<value_type>& target_data);

    void            set_uncertainty_transfer(const scm::piecewise_function_1d<value_type, float>& transfer_function);

protected:
    scm::piecewise_function_1d<value_type, float>  _uncertainty_transfer;

}; // class uncertainty_calculator_uncertainty_cast

} // namespace data
} // namespace scm

#include "uncertainty_calculator_uncertainty_cast.inl"

#endif // SCM_DATA_UNCERTAINTY_CALCULATOR_UNCERTAINTY_CAST_H_INCLUDED

