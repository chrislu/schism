
#ifndef SCM_MATH_INDICES_INL_INCLUDED
#define SCM_MATH_INDICES_INL_INCLUDED

#include <cassert>

namespace math
{
    template<unsigned dim>
    index_scalar_t get_linear_index(const math::vec<index_scalar_t, dim>& indices,
                                    const math::vec<index_scalar_t, dim>& dimensions)
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
    math::vec<index_scalar_t, dim> get_indices(index_scalar_t index,
                                               const math::vec<index_scalar_t, dim>& dimensions)
    {
        #ifdef _DEBUG
        assert(index < math::get_linear_index_end(dimensions));
        #endif

        math::vec<index_scalar_t, dim> tmp_indices;
        
        for (unsigned i = 0; i < dim; i++) {
            tmp_indices[i] = index % dimensions[i];
            index = index / dimensions[i];
        }

        return (tmp_indices);
    }

    template<unsigned dim>
    index_scalar_t get_linear_index_end(const math::vec<index_scalar_t, dim>& dimensions)
    {
        index_scalar_t max_index = 1;
        
        for (unsigned i = 0; i < dim; i++) {
            max_index *= dimensions[i];
        }

        return (max_index);
    }

} // namespace math

#endif // SCM_MATH_INDICES_INL_INCLUDED

