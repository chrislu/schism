
#ifndef REGULAR_GRID_DATA_3D_H_INCLUDED
#define REGULAR_GRID_DATA_3D_H_INCLUDED

#include <limits>

#include <boost/static_assert.hpp>
#include <boost/scoped_array.hpp>

#include <scm_core/math/math.h>
#include <scm_core/math/math_clamp_scal.h>
#include <scm_core/math/math_indices.h>

#include <data_analysis/regular_grid_data_properties.h>
#include <data_analysis/regular_grid_data_properties_write_accessor.h>

namespace scm
{
    template<typename> class regular_grid_data_3d_write_accessor;

    template<typename val_type>
    class regular_grid_data_3d : public regular_grid_data_properties_write_accessor<val_type>
    {
    public:
        typedef val_type                                value_type;
        typedef boost::scoped_array<val_type>           data_pointer_type;

    public:
        regular_grid_data_3d();
        virtual ~regular_grid_data_3d();

        const data_pointer_type&                        get_data() const;
        const regular_grid_data_properties<val_type>&   get_properties() const;

        void                                            update();

        const value_type& operator[](math::index_scalar_t index) const;
        const value_type& operator[](const math::index3d_t& index) const;

    protected:

        data_pointer_type                               _data;
        regular_grid_data_properties<val_type>          _properties;

    private:
        friend class regular_grid_data_3d_write_accessor<val_type>;

        // make sure of some things at compile time
        BOOST_STATIC_ASSERT(std::numeric_limits<val_type>::is_specialized);
    }; // regular_grid_data

} // namespace

#include "regular_grid_data_3d.inl"

#endif // REGULAR_GRID_DATA_3D_H_INCLUDED



