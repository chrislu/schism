
#ifndef MATH_VEC_H_INLCUDED
#define MATH_VEC_H_INLCUDED

//#define SCM_MATH_CORRECT_BIN_OPS

namespace scm {
namespace math {

template<typename scal_type, const unsigned dim>
class vec
{
    typedef scal_type   value_type;
}; // class vec<scm_scalar, dim>

} // namespace math
} // namespace scm

#include "vec.inl"

#endif // MATH_VEC_H_INLCUDED
