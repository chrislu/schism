
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>
#include <cassert>
#include <limits>

#include <scm/core/math/math.h>

namespace scm {
namespace data {

template<typename val_type,
         typename res_type>
piecewise_function_1d<val_type, res_type>::piecewise_function_1d()
  : _dirty(true)
{
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::insert_return_type
piecewise_function_1d<val_type, res_type>::add_stop(const stop_type& stop)
{
    _dirty = true;
    return (_function.insert(stop));
}

template<typename val_type,
         typename res_type>
void piecewise_function_1d<val_type, res_type>::del_stop(const stop_type& stop)
{
    del_stop(stop.first);
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::insert_return_type
piecewise_function_1d<val_type, res_type>::add_stop(value_type point, result_type value)
{
    return (add_stop(stop_type(point, value)));
}

template<typename val_type,
         typename res_type>
void piecewise_function_1d<val_type, res_type>::del_stop(value_type point)
{
    stop_iterator existent_stop = find_stop(point);

    if (existent_stop != _function.end()) {
        _function.erase(existent_stop);
        _dirty = true;
    }
}

template<typename val_type,
         typename res_type>
bool piecewise_function_1d<val_type, res_type>::dirty() const
{
    return (_dirty);
}

template<typename val_type,
         typename res_type>
void piecewise_function_1d<val_type, res_type>::dirty(const bool d)
{
    _dirty = d;
}

template<typename val_type,
         typename res_type>
void piecewise_function_1d<val_type, res_type>::clear()
{
    if (_function.size() != 0) {
        _dirty = true;
        _function.clear();
    }
}

template<typename val_type,
         typename res_type>
bool piecewise_function_1d<val_type, res_type>::empty() const
{
    return (_function.empty());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::result_type
piecewise_function_1d<val_type, res_type>::operator[](float point) const
{
    result_type result = result_type(0);

    const_stop_iterator lequal;
    const_stop_iterator gequal;

    if (std::numeric_limits<value_type>::is_integer){
        val_type v_min = (std::numeric_limits<value_type>::min)();
        val_type v_max = (std::numeric_limits<value_type>::max)();
        lequal = find_lequal_stop(static_cast<value_type>(scm::math::floor(scm::math::clamp<float>(point, v_min, v_max))));
        gequal = find_gequal_stop(static_cast<value_type>(scm::math::ceil(scm::math::clamp<float>(point, v_min, v_max))));
    }
    else {
        lequal = find_lequal_stop(static_cast<value_type>(point));
        gequal = find_gequal_stop(static_cast<value_type>(point));
    }

    if (lequal != _function.end() && gequal != _function.end()) {
        if (gequal->first == lequal->first) {
            result = lequal->second;
        }
        else {
            float     a = float(point - lequal->first) / float(gequal->first - lequal->first);
            result      = scm::math::lerp(lequal->second, gequal->second, a);
        }
    }

    return (result);
}

template<typename val_type,
         typename res_type>
std::size_t piecewise_function_1d<val_type, res_type>::num_stops() const
{
    return (_function.size());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::const_stop_iterator
piecewise_function_1d<val_type, res_type>::stops_begin() const
{
    return (_function.begin());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::const_stop_iterator
piecewise_function_1d<val_type, res_type>::stops_end() const
{
    return (_function.end());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::stop_iterator
piecewise_function_1d<val_type, res_type>::stops_begin()
{
    return (_function.begin());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::stop_iterator
piecewise_function_1d<val_type, res_type>::stops_end()
{
    return (_function.end());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::stop_iterator
piecewise_function_1d<val_type, res_type>::find_stop(value_type point)
{
    return (_function.find(point));
}

template<typename val_type,
        typename res_type>
typename piecewise_function_1d<val_type, res_type>::stop_iterator
piecewise_function_1d<val_type, res_type>::find_lequal_stop(value_type point)
{
    typename function_point_container_t::reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lequal_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::stop_iterator
piecewise_function_1d<val_type, res_type>::find_lesser_stop(value_type point)
{
    typename function_point_container_t::reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lesser_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::stop_iterator
piecewise_function_1d<val_type, res_type>::find_gequal_stop(value_type point)
{
    return (std::find_if(_function.begin(), _function.end(), gequal_op(point)));
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::stop_iterator
piecewise_function_1d<val_type, res_type>::find_greater_stop(value_type point)
{
    return (std::find_if(_function.begin(), _function.end(), greater_op(point)));
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::function_point_container_t::const_iterator
piecewise_function_1d<val_type, res_type>::find_lesser_stop(value_type point) const
{
    typename function_point_container_t::const_reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lesser_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::function_point_container_t::const_iterator
piecewise_function_1d<val_type, res_type>::find_lequal_stop(value_type point) const
{
    typename function_point_container_t::const_reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lequal_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::function_point_container_t::const_iterator
piecewise_function_1d<val_type, res_type>::find_gequal_stop(value_type point) const
{
    return (std::find_if(_function.begin(), _function.end(), gequal_op(point)));
}

template<typename val_type,
         typename res_type>
typename piecewise_function_1d<val_type, res_type>::function_point_container_t::const_iterator
piecewise_function_1d<val_type, res_type>::find_greater_stop(value_type point) const
{
    return (std::find_if(_function.begin(), _function.end(), greater_op(point)));
}

} // namespace data
} // namespace scm
