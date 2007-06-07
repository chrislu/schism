
#ifndef REGULAR_GRID_DATA_PROPERTIES_H_INCLUDED
#define REGULAR_GRID_DATA_PROPERTIES_H_INCLUDED

#include <data_analysis/value_range.h>

#include <scm/core/math/math.h>

namespace scm
{
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

        const scm::value_range<val_type>&       get_value_range() const;
        const math::vec<unsigned, 3>&           get_dimensions() const;
        const math::vec3f_t&                    get_spacing() const;
        const math::vec3f_t&                    get_origin() const;

    protected:
        scm::value_range<val_type>              _value_range;

        math::vec<unsigned, 3>                  _dimensions;
        math::vec3f_t                           _spacing;
        math::vec3f_t                           _origin;

    private:
        friend class regular_grid_data_properties_write_accessor<val_type>;
    };
} // namespace scm

#include "regular_grid_data_properties.inl"

#endif // REGULAR_GRID_DATA_PROPERTIES_H_INCLUDED



