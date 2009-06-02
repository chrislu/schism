
#include "plane.h"

#include <algorithm>

namespace scm {
namespace gl {

plane::plane()
  : _vector(plane::vec4_type(0.0f)),
    _p_corner(0u),
    _n_corner(0u)
{
}

plane::plane(const plane& p)
  : _vector(p._vector),
    _p_corner(p._p_corner),
    _n_corner(p._n_corner)
{
}

plane::plane(const plane::vec4_type& p)
  : _vector(p)
{
    normalize();
}

plane&
plane::operator=(const plane& rhs)
{
    plane tmp(rhs);
    swap(tmp);
    return (*this);
}

void
plane::swap(plane& rhs)
{
    std::swap(_vector,   rhs._vector);
    std::swap(_p_corner, rhs._p_corner);
    std::swap(_n_corner, rhs._n_corner);
}

const plane::vec3_type
plane::normal() const
{
    return (_vector);
}

plane::vec3_type::value_type
plane::distance(const plane::vec3_type& p) const
{
    return (  _vector.x * p.x
            + _vector.y * p.y
            + _vector.z * p.z
            + _vector.w);
}

const plane::vec4_type&
plane::vector() const
{
    return (_vector);
}

void
plane::transform(const plane::mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted(inverse(t));
}

void
plane::transform_preinverted(const plane::mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted_transposed(transpose(t));
}

void
plane::transform_preinverted_transposed(const plane::mat4_type& t)
{
    _vector = t * _vector;
    normalize();
}

unsigned
plane::p_corner() const
{
    return (_p_corner);
}

unsigned
plane::n_corner() const
{
    return (_n_corner);
}

void
plane::normalize()
{
    using namespace scm::math;
    vec4_type::value_type inv_len =   vec4_type::value_type(1)
                                    / sqrt(  _vector.x * _vector.x
                                           + _vector.y * _vector.y
                                           + _vector.z * _vector.z);
    _vector *= inv_len;
    update_corner_indices();
}

void
plane::update_corner_indices()
{
    _p_corner   = (  (_vector.x > 0.f ? 1u : 0)
                   | (_vector.y > 0.f ? 2u : 0)
                   | (_vector.z > 0.f ? 4u : 0));
    _n_corner   = (~_p_corner)&7;
}

} // namespace gl
} // namespace scm
