
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>
#include <cassert>
#include <limits>

namespace scm {
namespace data {
namespace detail {

template<typename T>
inline T non_linear_value_weight(typename boost::call_traits<T>::param_type val, float w)
{
    if (w >= 0.0f) {
        return (scm::math::pow(val, w + 1.0f));
    }
    else {
        return (1.0f - scm::math::abs(scm::math::pow(val - 1.0f, scm::math::round(-w + 1.0f))));//math::pow(val, 1.0f / (-w + 1.0f)));
    }
}

} //  namespace detail


template<typename val_type, typename scal_type>
piecewise_function_weighted_1d<val_type, scal_type>::piecewise_function_weighted_1d()
{
}

template<typename val_type, typename scal_type>
void piecewise_function_weighted_1d<val_type, scal_type>::add_point(val_type point, typename boost::call_traits<scal_type>::param_type value, float weight, val_type epsilon)
{
    typename function_point_container_t::iterator existent_point = find_point(point, epsilon);

    if (existent_point == _function.end()) {
        _function.insert(function_point_container_t::value_type(point, weighted_value(value, weight)));
    }
    else {
        existent_point->second._value  = value;
        existent_point->second._weight = weight;
    }
}

template<typename val_type, typename scal_type>
void piecewise_function_weighted_1d<val_type, scal_type>::del_point(val_type point, val_type epsilon)
{
    typename function_point_container_t::iterator existent_point = find_point(point, epsilon);

    if (existent_point != _function.end()) {
        _function.erase(existent_point);
    }
}
template<typename val_type, typename scal_type>
float piecewise_function_weighted_1d<val_type, scal_type>::get_point_weight(val_type point, val_type epsilon)
{
    typename function_point_container_t::iterator existent_point = find_point(point, epsilon);

    if (existent_point != _function.end()) {
        return (existent_point->second._weight);
    }
    else {
        return (0.0f);
    }
}

template<typename val_type, typename scal_type>
void piecewise_function_weighted_1d<val_type, scal_type>::clear()
{
    _function.clear();
}

template<typename val_type, typename scal_type>
bool piecewise_function_weighted_1d<val_type, scal_type>::empty() const
{
    return (_function.empty());
}

template<typename val_type, typename scal_type>
scal_type piecewise_function_weighted_1d<val_type, scal_type>::operator[](float point) const
{
    scal_type result = scal_type(0);

    typename  function_point_container_t::const_iterator lequal;
    typename  function_point_container_t::const_iterator gequal;

    if (std::numeric_limits<val_type>::is_integer){
        val_type v_min = (std::numeric_limits<val_type>::min)();
        val_type v_max = (std::numeric_limits<val_type>::max)();
        lequal = find_lequal_point(static_cast<val_type>(scm::math::floor(scm::math::clamp<float>(point, v_min, v_max))));
        gequal = find_gequal_point(static_cast<val_type>(scm::math::ceil(scm::math::clamp<float>(point, v_min, v_max))));
    }
    else {
        lequal = find_lequal_point(static_cast<val_type>(point));
        gequal = find_gequal_point(static_cast<val_type>(point));
    }

    if (lequal != _function.end() && gequal != _function.end()) {
        if (gequal->first == lequal->first) {
            result = lequal->second._value;
        }
        else {
            float     a = float(point - lequal->first) / float(gequal->first - lequal->first);
            a           = detail::non_linear_value_weight<float>(a, lequal->second._weight);
            result      = scm::math::lerp<scal_type>(lequal->second._value, gequal->second._value, a);
        }
    }

    return (result);
}

template<typename val_type, typename scal_type>
unsigned piecewise_function_weighted_1d<val_type, scal_type>::get_num_points() const
{
    return (_function.size());
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::const_point_iterator
    piecewise_function_weighted_1d<val_type, scal_type>::get_points_begin() const
{
    return (_function.begin());
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::const_point_iterator
    piecewise_function_weighted_1d<val_type, scal_type>::get_points_end() const
{
    return (_function.end());
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_point(val_type point, val_type epsilon)
{
    typename function_point_container_t::iterator result = _function.end();

    if (epsilon == val_type(0)) {
        result = _function.find(point);
    }
    else {
        typename function_point_container_t::iterator lequal     = find_lequal_point(point);
        typename function_point_container_t::iterator greater    = find_greater_point(point);

        if (lequal == _function.end() || greater == _function.end()) {
            if (lequal != greater) { // in case lequal == greater the container can only be empty
                if (lequal != _function.end()) {
                    if (scm::math::abs(point - lequal->first) <= epsilon) {
                        result = lequal;
                    }
                }
                else { // greater != _function.end()
                    if (scm::math::abs(greater->first - point) <= epsilon) {
                        result = greater;
                    }
                }
            }
        }
        else { // lequal != _function.end() && greater != _function.end()
            val_type g_eps = scm::math::abs(greater->first - point);
            val_type l_eps = scm::math::abs(point - lequal->first);
            
            if (g_eps <= epsilon && l_eps <= epsilon) {
                result = (g_eps < l_eps) ? greater : lequal;
            }
            else {
                if (l_eps <= epsilon) {
                    result = lequal;
                }
                else if (g_eps <= epsilon) {
                    result = greater;
                }
            }
        }
    }
    return (result);
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_lequal_point(val_type point)
{
    typename function_point_container_t::reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lequal_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_lesser_point(val_type point)
{
    typename function_point_container_t::reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lesser_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_gequal_point(val_type point)
{
    return (std::find_if(_function.begin(), _function.end(), gequal_op(point)));
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_greater_point(val_type point)
{
    return (std::find_if(_function.begin(), _function.end(), greater_op(point)));
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::const_iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_lesser_point(val_type point) const
{
    typename function_point_container_t::const_reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lesser_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::const_iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_lequal_point(val_type point) const
{
    typename function_point_container_t::const_reverse_iterator rit = std::find_if(_function.rbegin(), _function.rend(), lequal_op(point));

    return (rit == _function.rend() ? _function.end() : (++rit).base());
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::const_iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_gequal_point(val_type point) const
{
    return (std::find_if(_function.begin(), _function.end(), gequal_op(point)));
}

template<typename val_type, typename scal_type>
typename piecewise_function_weighted_1d<val_type, scal_type>::function_point_container_t::const_iterator
    piecewise_function_weighted_1d<val_type, scal_type>::find_greater_point(val_type point) const
{
    return (std::find_if(_function.begin(), _function.end(), greater_op(point)));
}

} // namespace data
} // namespace scm
