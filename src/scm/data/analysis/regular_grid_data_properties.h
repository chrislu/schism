
#ifndef SCM_DATA_REGULAR_GRID_DATA_PROPERTIES_H_INCLUDED
#define SCM_DATA_REGULAR_GRID_DATA_PROPERTIES_H_INCLUDED

#include <scm/data/analysis/value_range.h>

#include <scm/core/math/math.h>

namespace scm {
namespace data {

template<typename> class regular_grid_data_properties_write_accessor;

template<typename val_type>
class regular_grid_data_properties
{
public:
    typedef val_type                        value_type;

public:
    regular_grid_data_properties();
    regular_grid_data_properties(const regular_grid_data_properties& prop);
    virtual ~regular_grid_data_properties();

    regular_grid_data_properties<val_type>& operator =(const regular_grid_data_properties<val_type>& rhs);

    const scm::data::value_range<val_type>& get_value_range() const;
    const scm::math::vec<unsigned, 3>&      get_dimensions() const;
    const scm::math::vec3f&                 get_spacing() const;
    const scm::math::vec3f&                 get_origin() const;

protected:
    scm::data::value_range<val_type>        _value_range;

    scm::math::vec<unsigned, 3>             _dimensions;
    scm::math::vec3f                        _spacing;
    scm::math::vec3f                        _origin;

private:
    friend class regular_grid_data_properties_write_accessor<val_type>;
}; // class regular_grid_data_properties

} // namespace data
} // namespace scm

#include "regular_grid_data_properties.inl"

#endif // SCM_DATA_REGULAR_GRID_DATA_PROPERTIES_H_INCLUDED
