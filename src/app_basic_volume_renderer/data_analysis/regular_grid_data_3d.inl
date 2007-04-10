
#include <cassert>

#pragma warning (push)
#pragma warning (disable : 4146)

namespace scm
{
    template<typename val_type>
    regular_grid_data_3d<val_type>::regular_grid_data_3d()
    {
    }

    template<typename val_type>
    regular_grid_data_3d<val_type>::~regular_grid_data_3d()
    {

    }

    template<typename val_type>
    void regular_grid_data_3d<val_type>::update()
    {
        // reset value range
        scm::value_range<val_type> value_range;

        value_range.set_min((std::numeric_limits<val_type>::max)());
        value_range.set_max(std::numeric_limits<val_type>::is_integer ? (std::numeric_limits<val_type>::min)() : -(std::numeric_limits<val_type>::max)());

        for (math::index_scalar_t i = 0; i < math::get_linear_index_end(_properties.get_dimensions()); i++) {
            value_range.decide_and_set_min(_data[i]);
            value_range.decide_and_set_max(_data[i]);
        }

        set_value_range(_properties, value_range);
    }

    template<typename val_type>
    const typename regular_grid_data_3d<val_type>::data_pointer_type& regular_grid_data_3d<val_type>::get_data() const
    {
        return (_data);
    }

    template<typename val_type>
    const regular_grid_data_properties<val_type>& regular_grid_data_3d<val_type>::get_properties() const
    {
        return (_properties);
    }

    template<typename val_type>
    const val_type& regular_grid_data_3d<val_type>::operator[](unsigned index) const
    {
        assert(index < math::get_linear_index_end(_properties.get_dimensions()));
        
        return (_data[index]);
    }

    template<typename val_type>
    const val_type& regular_grid_data_3d<val_type>::operator[](const math::index3d_t& index) const
    {
        math::index3d_t clamp_index = math::clamp(index, math::index3d_t(0, 0, 0), _properties.get_dimensions() - math::vec<unsigned, 3>(1, 1, 1));

        return (_data[math::get_linear_index(clamp_index, _properties.get_dimensions())]);
    }

} // namespace scm

#pragma warning (pop)


