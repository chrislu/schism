

#include <cassert>

#include <scm/data/analysis/regular_grid_data_properties.h>

namespace scm {
namespace data {

template<typename val_type>
regular_grid_data_properties_write_accessor<val_type>::regular_grid_data_properties_write_accessor()
{
}

template<typename val_type>
regular_grid_data_properties_write_accessor<val_type>::~regular_grid_data_properties_write_accessor()
{
}

template<typename val_type>
void regular_grid_data_properties_write_accessor<val_type>::set_value_range(regular_grid_data_properties<val_type>& target_prop, const scm::data::value_range<val_type>& vr)
{
    target_prop._value_range = vr;
}

template<typename val_type>
void regular_grid_data_properties_write_accessor<val_type>::set_dimensions(regular_grid_data_properties<val_type>& target_prop, const scm::math::vec<unsigned, 3>& dim)
{
    target_prop._dimensions = dim;
}

template<typename val_type>
void regular_grid_data_properties_write_accessor<val_type>::set_spacing(regular_grid_data_properties<val_type>& target_prop, const scm::math::vec3f& spacing)
{
    target_prop._spacing = spacing;
}

template<typename val_type>
void regular_grid_data_properties_write_accessor<val_type>::set_origin(regular_grid_data_properties<val_type>& target_prop, const scm::math::vec3f& origin)
{
    target_prop._origin = origin;
}

} // namespace data
} // namespace scm
