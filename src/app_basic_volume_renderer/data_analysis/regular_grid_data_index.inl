
#include <cassert>

namespace scm
{
    template<unsigned dimension>
    regular_grid_data_index<dimension>::regular_grid_data_index(const math::vec<unsigned, dimension>& initial_index,
                                                                const math::vec<unsigned, dimension>& dimensions)
    : math::vec<unsigned, dimension>(initial_index),
      _linear_index(get_linear_index_from_indices(initial_index)),
      _dimensions(dimensions)
    {
    }

    template<unsigned dimension>
    regular_grid_data_index<dimension>::regular_grid_data_index(const regular_grid_data_index& i)
    : math::vec<unsigned, dimension>(i),
      _linear_index(get_linear_index_from_indices(i)),
      _dimensions(i._dimensions)
    {
    }
    
    template<unsigned dimension>
    regular_grid_data_index<dimension>& regular_grid_data_index<dimension>::operator=(const regular_grid_data_index<dimension>& rhs)
    {
        math::vec<unsigned, dimension>::operator =(rhs);
        _dimensions = rhs._dimensions;

        return (*this);
    }

    template<unsigned dimension>
    regular_grid_data_index<dimension>& regular_grid_data_index<dimension>::operator++()
    {
        math::vec<unsigned, dimension>::operator =(get_indices_from_linear_index(++_linear_index));
        return (*this);
    }

    template<unsigned dimension>
    regular_grid_data_index<dimension>& regular_grid_data_index<dimension>::operator--()
    {
        math::vec<unsigned, dimension>::operator =(get_indices_from_linear_index(--_linear_index));
        return (*this);
    }

    template<unsigned dimension>
    regular_grid_data_index<dimension> regular_grid_data_index<dimension>::operator++(int)
    {
        regular_grid_data_index<dimension> tmp = *this;

        math::vec<unsigned, dimension>::operator =(get_indices_from_linear_index(++_linear_index));

        return (tmp);
    }

    template<unsigned dimension>
    regular_grid_data_index<dimension> regular_grid_data_index<dimension>::operator--(int)
    {
        regular_grid_data_index<dimension> tmp = *this;

        math::vec<unsigned, dimension>::operator =(get_indices_from_linear_index(--_linear_index));
        
        return (tmp);
    }

    template<unsigned dimension>
    math::vec<unsigned, dimension> regular_grid_data_index<dimension>::get_indices_from_linear_index(unsigned index)
    {
        math::vec<unsigned, dimension> tmp_indices;
        
        #ifdef _DEBUG
        unsigned max_index = 1;
        for (unsigned i = 0; i < dimension; i++)
            max_index *= _dimensions[i];

        assert(index < max_index);
        #endif

        for (unsigned i = 0; i < dimension; i++) {
            tmp_indices[i] = index % _dimensions[i];
            index = index / _dimensions[i];
        }

        return (tmp_indices);
    }

    template<unsigned dimension>
    unsigned regular_grid_data_index<dimension>::get_linear_index_from_indices(const math::vec<unsigned, dimension>& indices)
    {
        unsigned tmp_index = indices[0];
        unsigned dim_amount = _dimensions[0];

        for (unsigned i = 1; i < dimension; i++) {
            tmp_index  += indices[i] * dim_amount;
            dim_amount *= _dimensions[i];
        }

        return (tmp_index);
    }

} // namespace scm



