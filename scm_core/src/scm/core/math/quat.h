
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_MATH_QUAT_H_INCLUDED
#define SCM_CORE_MATH_QUAT_H_INCLUDED

#include <scm/core/math/common.h>
#include <scm/core/math/vec3.h>
#include <scm/core/math/mat3.h>
#include <scm/core/math/mat4.h>

namespace scm {
namespace math {

template<typename scal_type>
class quat
{
public:
    typedef scal_type   value_type;

public:
    // ctors
    quat();
    quat(const value_type r, const value_type ii, const value_type ij, const value_type ik);
    quat(const value_type r, const vec<value_type, 3>& img);
    quat(const quat<value_type>& rhs);
    template<typename rhs_t> explicit quat(const quat<rhs_t>& rhs);

    void                            swap(quat<value_type>& rhs);

    quat<value_type>&               operator=(const quat<value_type>& rhs);
    template<typename rhs_t>
    quat<value_type>&               operator=(const quat<rhs_t>& rhs);

    quat<value_type>&               operator*=(const quat<value_type>& a);
    template<typename rhs_t>
    quat<value_type>&               operator*=(const quat<rhs_t>& a);

    const mat<value_type, 4, 4>     to_matrix() const;
    void                            retrieve_axis_angle(value_type& angle, vec<value_type, 3>& axis) const;

    static quat<value_type>         identity();
    static quat<value_type>         from_arc(const vec<value_type, 3>& a, const vec<value_type, 3>& b);
    static quat<value_type>         from_axis(const value_type angle, const vec<value_type, 3>& axis);
    static quat<value_type>         from_euler(const value_type pitch, const value_type yaw, const value_type roll);
    static quat<value_type>         from_matrix(const mat<value_type, 3, 3>& m);
    static quat<value_type>         from_matrix(const mat<value_type, 4, 4>& m);

public:
    value_type  w;
    union {value_type i; value_type x;};
    union {value_type j; value_type y;};
    union {value_type k; value_type z;};

}; // class quat

template<typename scal_type> const quat<scal_type>              operator*(const quat<scal_type>& lhs, const quat<scal_type>& rhs);
template<typename scal_l, typename scal_r> const quat<scal_l>   operator*(const quat<scal_l>& lhs, const quat<scal_r>& rhs);
template<typename scal_type> const vec<scal_type, 3>            operator*(const quat<scal_type>& lhs, const vec<scal_type, 3>& rhs);
template<typename scal_type> bool                               operator==(const quat<scal_type>& lhs, const quat<scal_type>& rhs);

template<typename scal_type> const quat<scal_type>      normalize(const quat<scal_type>& lhs);
template<typename scal_type> const quat<scal_type>      conjugate(const quat<scal_type>& lhs);
template<typename scal_type> const scal_type            magnitude(const quat<scal_type>& lhs);

template<typename scal_type> const quat<scal_type>      lerp(const quat<scal_type>& a, const quat<scal_type>& b, const scal_type u);
template<typename scal_type> const quat<scal_type>      slerp(const quat<scal_type>& a, const quat<scal_type>& b, const scal_type u);

} // namespace math
} // namespace scm

namespace std {

template<typename scal_type>
void swap(scm::math::quat<scal_type>& lhs,
          scm::math::quat<scal_type>& rhs);

} // namespace std


#include "quat.inl"

#endif // SCM_CORE_MATH_QUAT_H_INCLUDED
