
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <limits>

namespace scm {
namespace data {

template<typename value_type>
bool uncertainty_calculator_uncertainty_cast<value_type>::calculate_uncertainty_data(const scm::regular_grid_data_3d<value_type>& source_data,
                                               scm::regular_grid_data_3d<value_type>& target_data)
{
    if (   source_data.get_data().get() == 0
        || source_data.get_properties().get_dimensions().x == 0
        || source_data.get_properties().get_dimensions().y == 0
        || source_data.get_properties().get_dimensions().z == 0) {
        return (false);
    }

    scm::math::vec<unsigned, 3> dimensions  = source_data.get_properties().get_dimensions();
    value_type                  v_max       = (std::numeric_limits<value_type>::max)();
    bool                        is_int      = std::numeric_limits<value_type>::is_integer;
    float                       value_scale = static_cast<float>(v_max);

    try {
        get_data_ptr(target_data).reset(new scm::regular_grid_data_3d<value_type>::value_type[dimensions.x * dimensions.y * dimensions.z]);
    }
    catch (std::bad_alloc&) {
        get_data_ptr(target_data).reset();
        return (false);
    }

    set_dimensions(target_data, dimensions);

    scm::math::vec<unsigned, 3> index;
    for (index.x = 0; index.x < dimensions.x; ++index.x) {
        for (index.y = 0; index.y < dimensions.y; ++index.y) {
            float uncertainy_accum = 0.0f;
            for (index.z = dimensions.z - 1; index.z + 1 > 0; --index.z) {
                if (uncertainy_accum < 1.0) {
                    uncertainy_accum = scm::math::clamp(uncertainy_accum + _uncertainty_transfer[source_data[index]], 0.0f, 1.0f);
                }
                target_data[index] = static_cast<value_type>(value_scale*uncertainy_accum);
            }
        }
    }

    target_data.update();

    return (true);
}

template<typename value_type>
void uncertainty_calculator_uncertainty_cast<value_type>::set_uncertainty_transfer(const scm::piecewise_function_1d<value_type, float>& transfer_function)
{
    _uncertainty_transfer = transfer_function;
}

} // namespace data
} // namespace scm

