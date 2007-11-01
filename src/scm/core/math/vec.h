
#ifndef VEC_H_INLCUDED
#define VEC_H_INLCUDED

#include <cassert>

#include <boost/static_assert.hpp>

namespace math
{
    template<typename scm_scalar, unsigned dim>
    class vec
    {
    public:
        vec();
        explicit vec(const vec<scm_scalar, dim>& v);
        explicit vec(const scm_scalar s);

        vec<scm_scalar, dim>& operator=(const vec<scm_scalar, dim>& rhs);

        scm_scalar& operator[](const unsigned i);
        const scm_scalar& operator[](const unsigned i);

        scm_scalar  vec_array[dim];
    };

} // namespace math

#endif // VEC_H_INLCUDED



