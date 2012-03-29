
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <exception>
#include <stdexcept>

#include <boost/utility.hpp>

namespace scm {
namespace data {
namespace detail {

/*

template<typename val_type>
bool build_lookup_table<val_type>(boost::scoped_array<val_type>& dst,
                                  const piecewise_function_weighted_1d<unsigned char, val_type>& scal_trafu,
                                  unsigned size)
{
    if (size < 1) {
        return (false);
    }

    float    dst_ind_scal_factor = float(size - 1) / 255.0f;

    unsigned dst_ind_begin;
    unsigned dst_ind_end;

    float    dst_ind_begin_weight;
    val_type dst_ind_begin_value;
    val_type dst_ind_end_value;

    float lerp_factor;
    float lerp_factor_w;
    float part_step_size;

    // clear beginning
    if (scal_trafu.empty()) {
        dst_ind_begin = 0;
        dst_ind_end   = size - 1;
    }
    else {
        dst_ind_begin   = 0;
        dst_ind_end     = unsigned(math::floor(float(scal_trafu.stops_begin()->first) * dst_ind_scal_factor));
    }

    for (unsigned dst_ind = dst_ind_begin; dst_ind < dst_ind_end; ++dst_ind) {
        dst[dst_ind] = val_type(0);
    }

    // fill lookup table
    for (scm::piecewise_function_weighted_1d<unsigned char, val_type>::const_stop_iterator it_left = scal_trafu.stops_begin();
         it_left  != scal_trafu.stops_end();
         ++it_left) {

        dst_ind_begin           = unsigned(math::floor(float(it_left->first) * dst_ind_scal_factor));
        dst_ind_begin_value     = it_left->second._value;
        dst_ind_begin_weight    = it_left->second._weight;

        scm::piecewise_function_weighted_1d<unsigned char, val_type>::const_stop_iterator it_right = boost::next(it_left);
        if (it_right != scal_trafu.stops_end()) {
            dst_ind_end         = unsigned(math::floor(float(it_right->first) * dst_ind_scal_factor));
            dst_ind_end_value   = it_right->second._value;
        }
        else {
            dst_ind_end         = dst_ind_begin + 1;
            dst_ind_end_value   = dst_ind_begin_value;
        }
        
        part_step_size = 1.0f / float(dst_ind_end - dst_ind_begin);
        lerp_factor = 0.0f;
        lerp_factor_w = 0.0f;

        for (unsigned dst_ind = dst_ind_begin; dst_ind < dst_ind_end; ++dst_ind) {
            //lerp_factor = math::shoothstep(dst_ind_begin, dst_ind_end, dst_ind);
            //// smoothstep
            //float s = math::clamp(float(dst_ind-dst_ind_begin)/float(dst_ind_end-dst_ind_begin), 0.0f, 1.0f);
            //s = scm::detail::non_linear_value_weight<float>(s, dst_ind_begin_weight);
            //lerp_factor_w = (s*s*(3.0f-2.0f*s));

            lerp_factor_w = scm::detail::non_linear_value_weight<float>(lerp_factor, dst_ind_begin_weight);
            dst[dst_ind] = math::lerp(dst_ind_begin_value, dst_ind_end_value, lerp_factor_w);
            lerp_factor += part_step_size;
        }
    }

    // clear end
    for (unsigned dst_ind = dst_ind_end; dst_ind < size; ++dst_ind) {
        dst[dst_ind] = val_type(0);
    }


    // original code
    //float a;
    //float step = 255.0f / float(size - 1);
    //for (unsigned i = 0; i < size; i++) {
    //    a = float(i) * step;
    //    dst[i] = scal_trafu[a]; 
    //}

    return (true);
}

*/

template<typename val_type>
struct build_lookup_table_impl<val_type, unsigned char>
{
    static bool build_table(boost::scoped_array<val_type>& dst,
                            const piecewise_function_1d<unsigned char, val_type>& scal_trafu,
                            unsigned size)
    {
        using namespace scm::math;

        if (size < 1) {
            return (false);
        }
        
        float    dst_ind_scal_factor = float(size - 1) / 255.0f;
        
        unsigned dst_ind_begin;
        unsigned dst_ind_end;
        
        val_type dst_ind_begin_value;
        val_type dst_ind_end_value;
        
        float lerp_factor;
        float part_step_size;
        
        // clear beginning
        if (scal_trafu.empty()) {
            dst_ind_begin = 0;
            dst_ind_end   = size - 1;
        }
        else {
            dst_ind_begin   = 0;
            dst_ind_end     = unsigned(floor(float(scal_trafu.stops_begin()->first) * dst_ind_scal_factor));
        }
        
        for (unsigned dst_ind = dst_ind_begin; dst_ind < dst_ind_end; ++dst_ind) {
            dst[dst_ind] = val_type(0);
        }
        
        // fill lookup table
        for (typename scm::data::piecewise_function_1d<unsigned char, val_type>::const_stop_iterator it_left = scal_trafu.stops_begin();
            it_left  != scal_trafu.stops_end();
            ++it_left) {
        
            dst_ind_begin       = unsigned(floor(float(it_left->first) * dst_ind_scal_factor));
            dst_ind_begin_value = it_left->second;
        
            typename scm::data::piecewise_function_1d<unsigned char, val_type>::const_stop_iterator it_right = boost::next(it_left);
            if (it_right != scal_trafu.stops_end()) {
            dst_ind_end         = unsigned(floor(float(it_right->first) * dst_ind_scal_factor));
            dst_ind_end_value   = it_right->second;
            }
            else {
            dst_ind_end         = dst_ind_begin + 1;
            dst_ind_end_value   = dst_ind_begin_value;
            }
            
            part_step_size = 1.0f / float(dst_ind_end - dst_ind_begin);
            lerp_factor = 0.0f;
        
            for (unsigned dst_ind = dst_ind_begin; dst_ind < dst_ind_end; ++dst_ind) {
            //lerp_factor = math::shoothstep(dst_ind_begin, dst_ind_end, dst_ind);
            dst[dst_ind] = lerp(dst_ind_begin_value, dst_ind_end_value, lerp_factor);
            lerp_factor += part_step_size;
            }
        }
        
        // clear end
        for (unsigned dst_ind = dst_ind_end; dst_ind < size; ++dst_ind) {
            dst[dst_ind] = val_type(0);
        }
        
        
        // original code
        //float a;
        //float step = 255.0f / float(size - 1);
        //for (unsigned i = 0; i < size; i++) {
        //    a = float(i) * step;
        //    dst[i] = scal_trafu[a]; 
        //}
        
        return (true);
    }
}; // struct_look_uptable_impl

template<typename val_type>
struct build_lookup_table_impl<val_type, float>
{
    static bool build_table(boost::scoped_array<val_type>&                dst,
                            const piecewise_function_1d<float, val_type>& scal_trafu,
                            unsigned                                      size)
    {
        using namespace scm::math;

        if (size < 1) {
            return (false);
        }
    
        float    dst_ind_scal_factor = float(size - 1);
        
        unsigned dst_ind_begin;
        unsigned dst_ind_end;
        
        val_type dst_ind_begin_value;
        val_type dst_ind_end_value;
        
        float lerp_factor;
        float part_step_size;
        
        // clear beginning
        if (scal_trafu.empty()) {
            dst_ind_begin = 0;
            dst_ind_end   = size - 1;
        }
        else {
            dst_ind_begin   = 0;
            dst_ind_end     = unsigned(floor(scal_trafu.stops_begin()->first * dst_ind_scal_factor));
        }
        
        for (unsigned dst_ind = dst_ind_begin; dst_ind < dst_ind_end; ++dst_ind) {
            dst[dst_ind] = val_type(0);
        }
        
        // fill lookup table
        for (typename scm::data::piecewise_function_1d<float, val_type>::const_stop_iterator it_left = scal_trafu.stops_begin();
            it_left  != scal_trafu.stops_end();
            ++it_left) {
        
            dst_ind_begin       = unsigned(floor(it_left->first * dst_ind_scal_factor));
            dst_ind_begin_value = it_left->second;
        
            typename scm::data::piecewise_function_1d<float, val_type>::const_stop_iterator it_right = boost::next(it_left);
            if (it_right != scal_trafu.stops_end()) {
            dst_ind_end         = unsigned(floor(it_right->first * dst_ind_scal_factor));
            dst_ind_end_value   = it_right->second;
            }
            else {
            dst_ind_end         = dst_ind_begin + 1;
            dst_ind_end_value   = dst_ind_begin_value;
            }
            
            part_step_size = 1.0f / float(dst_ind_end - dst_ind_begin);
            lerp_factor = 0.0f;
        
            for (unsigned dst_ind = dst_ind_begin; dst_ind < dst_ind_end; ++dst_ind) {
            //lerp_factor = math::shoothstep(dst_ind_begin, dst_ind_end, dst_ind);
            dst[dst_ind] = lerp(dst_ind_begin_value, dst_ind_end_value, lerp_factor);
            lerp_factor += part_step_size;
            }
        }
        
        // clear end
        for (unsigned dst_ind = dst_ind_end; dst_ind < size; ++dst_ind) {
            dst[dst_ind] = val_type(0);
        }
        
        
        // original code
        //float a;
        //float step = 255.0f / float(size - 1);
        //for (unsigned i = 0; i < size; i++) {
        //    a = float(i) * step;
        //    dst[i] = scal_trafu[a]; 
        //}
        
        return (true);
    }
}; // struct_look_uptable_impl

} // namespace detail
} // namespace data
} // namespace scm
