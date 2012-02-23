
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <algorithm>

#include <scm/gl_core/primitives/box.h>

namespace scm {
namespace gl {

template<typename s>
plane_impl<s>::plane_impl()
  : _vector(vec4_type::zero()),
    _p_corner(0u),
    _n_corner(0u)
{
}

template<typename s>
plane_impl<s>::plane_impl(const plane_impl<s>& p)
  : _vector(p._vector),
    _p_corner(p._p_corner),
    _n_corner(p._n_corner)
{
}

template<typename s>
plane_impl<s>::plane_impl(const vec3_type& p0, const vec3_type& p1, const vec3_type& p2)
  : _p_corner(0u),
    _n_corner(0u)
{
    using namespace scm::math;

    // p2  p2 - p1
    // o<--------o p1
    //           |
    //           |  p0 - p1
    //           \/
    //           o p0

    vec3_type   n = math::normalize(cross(p2 - p1, p0 - p1));
    scal_type   d = -dot(n, p1);
    // describing plane equation
    //  - a*x + b*y + c*z + d = 0
    // hessesche normal form dot(n, p) - d = 0; so we use inverted d
    //  - so d gives you the distance of the origin related to the plane!

    _vector = vec4_type(n, d);
    update_corner_indices();
}

template<typename s>
plane_impl<s>::plane_impl(const vec4_type& p)
  : _vector(p)
{
    normalize();
}

template<typename s>
plane_impl<s>&
plane_impl<s>::operator=(const plane_impl<s>& rhs)
{
    plane_impl<s> tmp(rhs);
    swap(tmp);
    return (*this);
}

template<typename s>
void
plane_impl<s>::swap(plane_impl<s>& rhs)
{
    std::swap(_vector,   rhs._vector);
    std::swap(_p_corner, rhs._p_corner);
    std::swap(_n_corner, rhs._n_corner);
}

template<typename s>
const typename plane_impl<s>::vec3_type
plane_impl<s>::normal() const
{
    return (_vector);
}

template<typename s>
typename plane_impl<s>::scal_type
plane_impl<s>::distance() const
{
    return (-_vector.w);
}

template<typename s>
typename plane_impl<s>::scal_type
plane_impl<s>::distance(const vec3_type& p) const
{
    return (  _vector.x * p.x
            + _vector.y * p.y
            + _vector.z * p.z
            + _vector.w);
}

template<typename s>
const typename plane_impl<s>::vec4_type&
plane_impl<s>::vector() const
{
    return (_vector);
}

template<typename s>
void
plane_impl<s>::reverse()
{
    _vector *= scal_type(-1);
}

template<typename s>
void
plane_impl<s>::transform(const mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted(inverse(t));
}

template<typename s>
void
plane_impl<s>::transform_preinverted(const mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted_transposed(transpose(t));
}

template<typename s>
void
plane_impl<s>::transform_preinverted_transposed(const mat4_type& t)
{
    _vector = t * _vector;
    normalize();
}

template<typename s>
unsigned
plane_impl<s>::p_corner() const
{
    return (_p_corner);
}

template<typename s>
unsigned
plane_impl<s>::n_corner() const
{
    return (_n_corner);
}

template<typename s>
typename plane_impl<s>::classification_result
plane_impl<s>::classify(const box_type& b, scal_type e) const
{
    using namespace scm::math;

    if (distance(b.corner(_n_corner)) > e) {
        return (front);
    }
    else if (distance(b.corner(_p_corner)) > e) {
        return (intersecting);
    }
    else {
        return (back);
    }
}

template<typename s>
typename plane_impl<s>::classification_result
plane_impl<s>::classify(const rect_type& b, scal_type e) const
{
    using namespace scm::math;

    if (distance(b.corner(_n_corner)) > e) {
        return (front);
    }
    else if (distance(b.corner(_p_corner)) > e) {
        return (intersecting);
    }
    else {
        return (back);
    }
}

template<typename s>
typename plane_impl<s>::classification_result
plane_impl<s>::classify(const vec3_type& p, scal_type e) const
{
    using namespace scm::math;

    scal_type d = distance(p);

    if (d > e) {
        return (front);
    }
    else if (d < -e) {
        return (back);
    }
    else {
        return (coinciding);
    }
}

template<typename s>
bool
plane_impl<s>::intersect(const ray_type& r, vec3_type& hit, scal_type e) const
{
    using namespace scm::math;

    vec3_type  n      = normal();
    scal_type  dot_nd = dot(n, r.direction());
    if (abs(dot_nd) < e) {
        return (false); // we are parallel to the plane
    }
    scal_type  dot_no = dot(n, r.origin());
    scal_type  t      = -(_vector.w + dot_no) / dot_nd;

    // calculate the intersection point for return
    hit = r.origin() + t * r.direction();

    return (t > scal_type(0));
}

template<typename s>
void
plane_impl<s>::normalize()
{
    using namespace scm::math;
    typename vec4_type::value_type inv_len = typename  vec4_type::value_type(1)
                                                     / sqrt(  _vector.x * _vector.x
                                                            + _vector.y * _vector.y
                                                            + _vector.z * _vector.z);
    _vector *= inv_len;
    update_corner_indices();
}

template<typename s>
void
plane_impl<s>::update_corner_indices()
{
    _p_corner   = (  (_vector.x > 0.f ? 1u : 0)
                   | (_vector.y > 0.f ? 2u : 0)
                   | (_vector.z > 0.f ? 4u : 0));
    _n_corner   = (~_p_corner)&7;
}

} // namespace gl
} // namespace scm
