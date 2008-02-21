
#ifndef DATA_REGULAR_GRID_INDICES_H_INCLUDED
#define DATA_REGULAR_GRID_INDICES_H_INCLUDED

#include <limits>

#include <scm/core/math/math.h>

namespace scm {
namespace data {

typedef unsigned                      index_scalar_t;

//const index_scalar_t                  illegal_index = index_scalar_t(0) - 1; // (std::numeric_limits<index_scalar_t>::max)()

typedef scm::math::vec<index_scalar_t, 2>  index2d_t;
typedef scm::math::vec<index_scalar_t, 3>  index3d_t;

template<unsigned dim>
index_scalar_t get_linear_index(const scm::math::vec<index_scalar_t, dim>& indices,
                                const scm::math::vec<index_scalar_t, dim>& dimensions);

template<unsigned dim>
scm::math::vec<index_scalar_t, dim> get_indices(index_scalar_t index,
                                                const scm::math::vec<index_scalar_t, dim>& dimensions);

// get the maximum index from a dimensions vector
// returns just one behind the max index (like end()), so 0 points out illegal index
template<unsigned dim>
index_scalar_t get_linear_index_end(const scm::math::vec<index_scalar_t, dim>& dimensions);

} // namespace data
} // namespace scm

#include "regular_grid_indices.inl"

#endif // DATA_REGULAR_GRID_INDICES_H_INCLUDED



