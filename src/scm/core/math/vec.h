
#ifndef MATH_VEC_H_INLCUDED
#define MATH_VEC_H_INLCUDED

#include <algorithm>

namespace scm {
namespace math {

template<typename scal_type, const unsigned dim>
class vec
{
    typedef scal_type   value_type;
}; // class vec<scm_scalar, dim>

} // namespace math
} // namespace scm

namespace std {

template<typename scal_type, const unsigned dim>
void swap(scm::math::vec<scal_type, dim>& lhs,
          scm::math::vec<scal_type, dim>& rhs);

} // namespace std

#include "vec.inl"

#endif // MATH_VEC_H_INLCUDED
