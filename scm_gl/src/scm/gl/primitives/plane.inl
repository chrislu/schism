
#include <algorithm>

#include <scm/gl/primitives/box.h>

namespace scm {
namespace gl {

template<typename s>
plane_impl<s>::plane_impl()
  : _vector(typename plane_impl<s>::vec4_type::zero()),
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
plane_impl<s>::plane_impl(const typename plane_impl<s>::vec4_type& p)
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
plane_impl<s>::distance(const typename plane_impl<s>::vec3_type& p) const
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
plane_impl<s>::transform(const typename plane_impl<s>::mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted(inverse(t));
}

template<typename s>
void
plane_impl<s>::transform_preinverted(const typename plane_impl<s>::mat4_type& t)
{
    using namespace scm::math;
    transform_preinverted_transposed(transpose(t));
}

template<typename s>
void
plane_impl<s>::transform_preinverted_transposed(const typename plane_impl<s>::mat4_type& t)
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
plane_impl<s>::classify(const typename plane_impl<s>::box_type& b) const
{
    using namespace scm::math;

    if (distance(b.corner(_n_corner)) > vec3_type::value_type(0)) {
        return (front);
    }
    else if (distance(b.corner(_p_corner)) > vec3_type::value_type(0)) {
        return (intersect);
    }
    else {
        return (back);
    }
}

template<typename s>
void
plane_impl<s>::normalize()
{
    using namespace scm::math;
    vec4_type::value_type inv_len =   vec4_type::value_type(1)
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
