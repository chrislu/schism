
#ifndef SCM_DATA_REGULAR_GRID_DATA_PROPERTIES_WRITE_ACCESSOR_H_INCLUDED
#define SCM_DATA_REGULAR_GRID_DATA_PROPERTIES_WRITE_ACCESSOR_H_INCLUDED

namespace scm {
namespace data {

template<typename> class regular_grid_data_properties;

template<typename val_type>
class regular_grid_data_properties_write_accessor
{
public:
    virtual ~regular_grid_data_properties_write_accessor();

protected:
    virtual void set_value_range(regular_grid_data_properties<val_type>& target_prop, const scm::data::value_range<val_type>& vr);
    virtual void set_dimensions(regular_grid_data_properties<val_type>& target_prop,  const math::vec<unsigned, 3>& dim);
    virtual void set_spacing(regular_grid_data_properties<val_type>& target_prop,     const math::vec3f_t& spacing);
    virtual void set_origin(regular_grid_data_properties<val_type>& target_prop,      const math::vec3f_t& origin);

    regular_grid_data_properties_write_accessor();
    regular_grid_data_properties_write_accessor(const regular_grid_data_properties_write_accessor<val_type>& s); // no implementation

}; // class regular_grid_data_properties_write_accessor

} // namespace data
} // namespace scm

#include "regular_grid_data_properties_write_accessor.inl"

#endif // SCM_DATA_REGULAR_GRID_DATA_PROPERTIES_WRITE_ACCESSOR_H_INCLUDED
