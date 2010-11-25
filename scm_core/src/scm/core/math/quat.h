
#ifndef SCM_CORE_MATH_QUAT_H_INCLUDED
#define SCM_CORE_MATH_QUAT_H_INCLUDED

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

template<typename scal_type> const quat<scal_type>      normalize(const quat<scal_type>& lhs);
template<typename scal_type> const quat<scal_type>      conjugate(const quat<scal_type>& lhs);
template<typename scal_type> const scal_type            magnitude(const quat<scal_type>& lhs);


} // namespace math
} // namespace scm

#include "quat.inl"

#endif // SCM_CORE_MATH_QUAT_H_INCLUDED
