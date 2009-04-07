
#ifndef SCM_DATA_REGULAR_GRID_DATA_INDEX_H_INCLUDED
#define SCM_DATA_REGULAR_GRID_DATA_INDEX_H_INCLUDED

#include <boost/static_assert.hpp>

#include <scm/core/math/math.h>

namespace scm {
namespace data {

template<unsigned dimension>
class regular_grid_data_index : public scm::math::vec<unsigned, dimension>
{
public:
    regular_grid_data_index(const regular_grid_data_index& i);
    regular_grid_data_index& operator=(const regular_grid_data_index<dimension>& rhs);

    regular_grid_data_index<dimension>&         operator++();
    regular_grid_data_index<dimension>&         operator--();
    regular_grid_data_index<dimension>          operator++(int);
    regular_grid_data_index<dimension>          operator--(int);

protected:
    unsigned                                    _linear_index;

    const math::vec<unsigned, dimension>&       _dimensions;

    scm::math::vec<unsigned, dimension>         get_indices_from_linear_index(unsigned index);
    unsigned                                    get_linear_index_from_indices(const scm::math::vec<unsigned, dimension>& indices);

private:
    regular_grid_data_index(const scm::math::vec<unsigned, dimension>& initial_index,
                            const scm::math::vec<unsigned, dimension>& dimensions);

    template <typename val_type, unsigned dimension> friend class regular_grid_data;

    // make sure of some things at compile time
    BOOST_STATIC_ASSERT(dimension > 0 && dimension < 4);
}; // regular_grid_data_iterator

} // namespace data
} // namespace scm

#include "regular_grid_data_index.inl"

#endif // SCM_DATA_REGULAR_GRID_DATA_INDEX_H_INCLUDED
