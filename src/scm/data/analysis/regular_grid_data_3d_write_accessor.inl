
#include <cassert>

namespace scm {
namespace data {

template<typename val_type>
regular_grid_data_3d_write_accessor<val_type>::regular_grid_data_3d_write_accessor()
{
}

template<typename val_type>
regular_grid_data_3d_write_accessor<val_type>::~regular_grid_data_3d_write_accessor()
{
}

template<typename val_type>
regular_grid_data_3d_write_accessor<val_type>::regular_grid_data_3d_write_accessor(const regular_grid_data_3d_write_accessor<val_type>& s)
{
    assert(0);
}

template<typename val_type>
void regular_grid_data_3d_write_accessor<val_type>::update(regular_grid_data_3d<val_type>& target_data)
{
    target_data.update();
}

template<typename val_type>
typename regular_grid_data_3d<val_type>::data_pointer_type& regular_grid_data_3d_write_accessor<val_type>::get_data_ptr(regular_grid_data_3d<val_type>& target_data)
{
    return target_data._data;
}

template<typename val_type>
void regular_grid_data_3d_write_accessor<val_type>::set_value_range(regular_grid_data_3d<val_type>& target_data, const scm::data::value_range<val_type>& vr)
{
    target_data.set_value_range(target_data._properties, vr);
}

template<typename val_type>
void regular_grid_data_3d_write_accessor<val_type>::set_dimensions(regular_grid_data_3d<val_type>& target_data, const scm::math::vec<unsigned, 3>& dim)
{
    target_data.set_dimensions(target_data._properties, dim);
}

template<typename val_type>
void regular_grid_data_3d_write_accessor<val_type>::set_spacing(regular_grid_data_3d<val_type>& target_data, const scm::math::vec3f& spacing)
{
    target_data.set_spacing(target_data._properties, spacing);
}

template<typename val_type>
void regular_grid_data_3d_write_accessor<val_type>::set_origin(regular_grid_data_3d<val_type>& target_data, const scm::math::vec3f& origin)
{
    target_data.set_origin(target_data._properties, origin);
}

} // namespace data
} // namespace scm
