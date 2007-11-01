
#ifndef MATH_INDICES_H_INCLUDED
#define MATH_INDICES_H_INCLUDED

#include <limits>

#include <scm/core/math/math_clamp_scal.h>
#include <scm/core/math/math_vec.h>

namespace math
{
    typedef unsigned                      index_scalar_t;

    //const index_scalar_t                  illegal_index = index_scalar_t(0) - 1; // (std::numeric_limits<index_scalar_t>::max)()

    typedef math::vec<index_scalar_t, 2>  index2d_t;
    typedef math::vec<index_scalar_t, 3>  index3d_t;

    template<unsigned dim>
    index_scalar_t get_linear_index(const math::vec<index_scalar_t, dim>& indices,
                                    const math::vec<index_scalar_t, dim>& dimensions);

    template<unsigned dim>
    math::vec<index_scalar_t, dim> get_indices(index_scalar_t index,
                                               const math::vec<index_scalar_t, dim>& dimensions);

    // get the maximum index from a dimensions vector
    // returns just one behind the max index (like end()), so 0 points out illegal index
    template<unsigned dim>
    index_scalar_t get_linear_index_end(const math::vec<index_scalar_t, dim>& dimensions);

} // namespace math

#include "math_indices.inl"

#endif // MATH_INDICES_H_INCLUDED



