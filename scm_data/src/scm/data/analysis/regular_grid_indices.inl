
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <cassert>

namespace scm {
namespace data {

template<unsigned dim>
index_scalar_t get_linear_index(const scm::math::vec<index_scalar_t, dim>& indices,
                                const scm::math::vec<index_scalar_t, dim>& dimensions)
{
    #ifdef _DEBUG
    for (unsigned i = 0; i < dim; i++) {
        assert(indices[i] < dimensions[i]);
    }
    #endif

    index_scalar_t tmp_index  = indices[0];
    index_scalar_t dim_amount = dimensions[0];

    for (unsigned i = 1; i < dim; i++) {
        tmp_index  += indices[i] * dim_amount;
        dim_amount *= dimensions[i];
    }

    return (tmp_index);
}

template<unsigned dim>
scm::math::vec<index_scalar_t, dim> get_indices(index_scalar_t index,
                                                const scm::math::vec<index_scalar_t, dim>& dimensions)
{
    #ifdef _DEBUG
    assert(index < math::get_linear_index_end(dimensions));
    #endif

    scm::math::vec<index_scalar_t, dim> tmp_indices;
    
    for (unsigned i = 0; i < dim; i++) {
        tmp_indices[i] = index % dimensions[i];
        index = index / dimensions[i];
    }

    return (tmp_indices);
}

template<unsigned dim>
index_scalar_t get_linear_index_end(const scm::math::vec<index_scalar_t, dim>& dimensions)
{
    index_scalar_t max_index = 1;
    
    for (unsigned i = 0; i < dim; ++i) {
        max_index *= dimensions[i];
    }

    return (max_index);
}
} // namespace data
} // namespace scm

