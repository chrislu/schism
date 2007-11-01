
#ifndef SCM_DATA_REGULAR_GRID_DATA_3D_WRITE_ACCESSOR_H_INCLUDED
#define SCM_DATA_REGULAR_GRID_DATA_3D_WRITE_ACCESSOR_H_INCLUDED

namespace scm {
namespace data {

template<typename val_type>
class regular_grid_data_3d_write_accessor
{
public:
    virtual ~regular_grid_data_3d_write_accessor();

protected:
    regular_grid_data_3d_write_accessor();
    regular_grid_data_3d_write_accessor(const regular_grid_data_3d_write_accessor<val_type>& s); // no implementation

    // todo: remove the following
    virtual void update(regular_grid_data_3d<val_type>& target_data);

    virtual void set_value_range(regular_grid_data_3d<val_type>& target_data, const scm::data::value_range<val_type>& vr);
    virtual void set_dimensions(regular_grid_data_3d<val_type>& target_data,  const math::vec<unsigned, 3>& dim);
    virtual void set_spacing(regular_grid_data_3d<val_type>& target_data,     const math::vec3f_t& spacing);
    virtual void set_origin(regular_grid_data_3d<val_type>& target_data,      const math::vec3f_t& origin);

    typename regular_grid_data_3d<val_type>::data_pointer_type& get_data_ptr(regular_grid_data_3d<val_type>& target_data);

}; // regular_grid_data_3d_writer

} // namespace data
} // namespace scm

#include "regular_grid_data_3d_write_accessor.inl"

#endif // SCM_DATA_REGULAR_GRID_DATA_3D_WRITE_ACCESSOR_H_INCLUDED
