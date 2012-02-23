
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

namespace scm {
namespace gl {

template<typename s>
ray_impl<s>::ray_impl()
  : _origin(typename ray_impl::vec3_type::value_type(0)),
    _direction(typename ray_impl::vec3_type::value_type(0))
{
}

template<typename s>
ray_impl<s>::ray_impl(const ray_impl<s>& p)
  : _origin(p._origin),
    _direction(p._direction)
{
}

template<typename s>
ray_impl<s>::ray_impl(const typename ray_impl<s>::vec3_type& org,
                      const typename ray_impl::vec3_type& dir)
  : _origin(org),
    _direction(dir)
{
    normalize();
}

template<typename s>
ray_impl<s>&
ray_impl<s>::operator=(const ray_impl<s>& rhs)
{
    ray_impl<s> tmp(rhs);
    swap(tmp);
    return (*this);
}

template<typename s>
void
ray_impl<s>::swap(ray_impl<s>& rhs)
{
    std::swap(_origin,    rhs._origin);
    std::swap(_direction, rhs._direction);
}

template<typename s>
void
ray_impl<s>::transform(const typename ray_impl<s>::mat4_type& t)
{
    using namespace scm::math;

    //mat4_type inv_trans = transpose(inverse(t));

    //_direction = vec3_type(inv_trans * vec4_type(_direction, typename vec4_type::value_type(0)));
    _direction = vec3_type(t * vec4_type(_direction, typename vec4_type::value_type(0)));
    _origin    = vec3_type(t * vec4_type(_origin,    typename vec4_type::value_type(1)));

    normalize();
}

template<typename s>
void
ray_impl<s>::transform_preinverted(const typename ray_impl<s>::mat4_type& it)
{
    using namespace scm::math;

    //mat4_type inv_trans = transpose(it);

    //_direction = vec3_type(inv_trans   * vec4_type(_direction, typename vec4_type::value_type(0)));
    _direction = vec3_type(inverse(it) * vec4_type(_direction, typename vec4_type::value_type(0)));
    _origin    = vec3_type(inverse(it) * vec4_type(_origin,    typename vec4_type::value_type(1)));

    normalize();
}

template<typename s>
const typename ray_impl<s>::vec3_type&
ray_impl<s>::origin() const
{
    return (_origin);
}

template<typename s>
const typename ray_impl<s>::vec3_type&
ray_impl<s>::direction() const
{
    return (_direction);
}

template<typename s>
void
ray_impl<s>::normalize()
{
    _direction = math::normalize(_direction);
}

} // namespace gl
} // namespace scm
