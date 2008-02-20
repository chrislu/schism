
#ifndef VEC_H_INLCUDED
#define VEC_H_INLCUDED

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

#endif // VEC_H_INLCUDED
