
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_BUILD_LOOKUPTABLE_H_INCLUDED
#define SCM_GL_UTIL_BUILD_LOOKUPTABLE_H_INCLUDED

#include <boost/scoped_array.hpp>

#include <scm/gl_util/data/analysis/transfer_function/piecewise_function_1d.h>
#include <scm/gl_util/data/analysis/transfer_function/piecewise_function_weighted_1d.h>

namespace scm {
namespace data {

namespace detail {

template<typename val_type,
         typename inp_type>
struct build_lookup_table_impl
{
    static bool build_table(boost::scoped_array<val_type>& dst,
                            const piecewise_function_1d<inp_type, val_type>& scal_trafu,
                            unsigned table_size);

}; // struct build_lookup_table_impl

} // namespace detail

template<typename val_type,
         typename inp_type>
bool build_lookup_table(boost::scoped_array<val_type>& dst,
                        const piecewise_function_1d<inp_type, val_type>& scal_trafu,
                        unsigned table_size)
{
    return (detail::build_lookup_table_impl<val_type, inp_type>::build_table(dst, scal_trafu, table_size));
}

/*
template<typename val_type>
bool build_lookup_table(boost::scoped_array<val_type>& dst, const piecewise_function_1d<unsigned char, val_type>& scal_trafu, unsigned size);
template<typename val_type>
bool build_lookup_table(boost::scoped_array<val_type>& dst, const piecewise_function_1d<float, val_type>& scal_trafu, unsigned size);
template<typename val_type>
bool build_lookup_table(boost::scoped_array<val_type>& dst, const piecewise_function_weighted_1d<unsigned char, val_type>& scal_trafu, unsigned size);
*/

//template<typename val_type>
//bool build_preintegrated_lookup_table(boost::scoped_array<val_type>& dst,
//                                      const piecewise_function_1d<unsigned char, val_type>& scal_trafu,
//                                      unsigned size);

} // namespace data
} // namespace scm

#include "build_lookup_table.inl"

#endif // SCM_GL_UTIL_BUILD_LOOKUPTABLE_H_INCLUDED

