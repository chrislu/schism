
#ifndef SCM_DATA_BUILD_LOOKUPTABLE_H_INCLUDED
#define SCM_DATA_BUILD_LOOKUPTABLE_H_INCLUDED

#include <boost/scoped_array.hpp>

#include <scm/data/analysis/transfer_function/piecewise_function_1d.h>
#include <scm/data/analysis/transfer_function/piecewise_function_weighted_1d.h>

namespace scm {
namespace data {
    
template<typename val_type>
bool build_lookup_table(boost::scoped_array<val_type>& dst, const piecewise_function_1d<unsigned char, val_type>& scal_trafu, unsigned size);
template<typename val_type>
bool build_lookup_table(boost::scoped_array<val_type>& dst, const piecewise_function_weighted_1d<unsigned char, val_type>& scal_trafu, unsigned size);

//template<typename val_type>
//bool build_preintegrated_lookup_table(boost::scoped_array<val_type>& dst,
//                                      const piecewise_function_1d<unsigned char, val_type>& scal_trafu,
//                                      unsigned size);

} // namespace data
} // namespace scm

#include "build_lookup_table.inl"

#endif // SCM_DATA_BUILD_LOOKUPTABLE_H_INCLUDED

