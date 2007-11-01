
#include <limits>

#include <scm/core/math/math.h>

//#pragma warning (push)
//#pragma warning (disable : 4146)

namespace scm {
namespace data {

bool histogram_1d_calculator<unsigned char>::calculate(scm::histogram_1d<unsigned char>& histogram,
                                                       const scm::regular_grid_data_3d<unsigned char>& data,
                                                       unsigned num_bins)
{
    if (   data.get_data().get() == 0
        || data.get_properties().get_dimensions().x == 0
        || data.get_properties().get_dimensions().y == 0
        || data.get_properties().get_dimensions().z == 0) {
        return (false);
    }

    unsigned char v_min         = (std::numeric_limits<unsigned char>::min)();
    unsigned char v_max         = (std::numeric_limits<unsigned char>::max)();
    int           v_range       = v_max - v_min;
    int           v_magnitude   = v_max - v_min + 1;

    // if number of bins not set generate them
    if (num_bins == 0) {
        num_bins = unsigned(10.0 * math::log10(double(v_magnitude)));
    }

    histogram._bins.resize(num_bins);

    // create histogram
    scm::histogram_1d<unsigned char>::bin_container_t::size_type  bin_index;
    int value;

    for (unsigned i = 0; i < math::get_linear_index_end(data.get_properties().get_dimensions()); i++) {
        value       = data.get_data()[i];
        bin_index   = (unsigned)math::floor(float(value - v_min)/float(v_magnitude) * (num_bins));

        histogram._bins[bin_index]._absolute_amount++;
    }

    unsigned num_voxels =   data.get_properties().get_dimensions().x
                          * data.get_properties().get_dimensions().y
                          * data.get_properties().get_dimensions().z;
    // calculate relative amounts and set value_ranges
    float bin_factor = (float)v_magnitude / (float)(num_bins + v_min);

    histogram._max_relative_amount = 0.0f;
    histogram._max_absolute_amount = 0;

    for (unsigned bi = 0; bi < num_bins; bi++) {
        histogram._bins[bi]._relative_amount = (float)histogram._bins[bi]._absolute_amount / (float)num_voxels;

        histogram._max_relative_amount = math::max(histogram._max_relative_amount,
                                                   histogram._bins[bi]._relative_amount);

        histogram._max_absolute_amount = math::max(histogram._max_absolute_amount,
                                                   histogram._bins[bi]._absolute_amount);

        unsigned char min = (unsigned char)math::ceil((float)bi * bin_factor);
        unsigned char max = (unsigned char)math::ceil((float)(bi+1) * bin_factor) - 1;

        histogram._bins[bi]._value_range.set_min(min);
        histogram._bins[bi]._value_range.set_max(max);
    }

    return (true);
}

} // namespace data
} // namespace scm

//#pragma warning (pop)

