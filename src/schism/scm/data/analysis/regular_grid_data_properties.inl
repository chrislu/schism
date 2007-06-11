
#include <cassert>

namespace scm {
namespace data {

template<typename val_type>
regular_grid_data_properties<val_type>::regular_grid_data_properties()
    : _dimensions(0, 0, 0),
      _spacing(1.0f, 1.0f, 1.0f),
      _origin(0.0f, 0.0f, 0.0f)
{
}

template<typename val_type>
regular_grid_data_properties<val_type>::regular_grid_data_properties(const regular_grid_data_properties& prop)
    : _dimensions(prop._dimensions),
      _spacing(prop._spacing),
      _origin(prop._origin),
      _value_range(prop._value_range)
{
}

template<typename val_type>
regular_grid_data_properties<val_type>::~regular_grid_data_properties()
{
}

template<typename val_type>
regular_grid_data_properties<val_type>& regular_grid_data_properties<val_type>::operator =(const regular_grid_data_properties<val_type>& rhs)
{
    _dimensions     = rhs._dimensions;
    _spacing        = rhs._spacing;
    _origin         = rhs._origin;
    _value_range    = rhs._value_range;

    return (*this);
}

template<typename val_type>
const scm::data::value_range<val_type>& regular_grid_data_properties<val_type>::get_value_range() const
{
    return (_value_range);
}

template<typename val_type>
const math::vec<unsigned, 3>& regular_grid_data_properties<val_type>::get_dimensions() const
{
    return (_dimensions);
}

template<typename val_type>
const math::vec3f_t& regular_grid_data_properties<val_type>::get_spacing() const
{
    return (_spacing);
}

template<typename val_type>
const math::vec3f_t& regular_grid_data_properties<val_type>::get_origin() const
{
    return (_origin);
}

} // namespace data
} // namespace scm
