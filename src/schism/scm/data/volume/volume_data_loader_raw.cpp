
#include "volume_data_loader_raw.h"

#include <exception>
#include <stdexcept>

//#pragma warning (disable : 4561 4793)
#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/spirit/core.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/data/analysis/regular_grid_data_3d_write_accessor.h>

namespace {

struct set_uint
{
    set_uint(unsigned& i) : _i(i) {}
    unsigned& _i;

    void operator()(unsigned i) const {
        _i = i;
    }
};

bool parse_raw_file_name(const std::string& filename,
                         unsigned& width,
                         unsigned& height,
                         unsigned& depth,
                         unsigned& num_components,
                         unsigned& bit_per_voxel)
{
    using namespace boost::spirit;

    typedef std::string::const_iterator        iterator_t;

    rule<scanner<iterator_t> > infos =    str_p("_w") >> uint_p[set_uint(width)] 
                                       >> str_p("_h") >> uint_p[set_uint(height)]
                                       >> str_p("_d") >> uint_p[set_uint(depth)]
                                       >> str_p("_c") >> uint_p[set_uint(num_components)]
                                       >> str_p("_b") >> uint_p[set_uint(bit_per_voxel)]; 

    rule<scanner<iterator_t> > r = +((infos >> str_p(".raw")) | (*ch_p("_") >> (anychar_p - ch_p("_"))));

    iterator_t first = filename.begin();
    iterator_t last  = filename.end();


    parse_info<iterator_t> info = parse(first, last, r);

    if (info.full)
        return (true);
    else
        return (false);
}

} // namespace

using namespace scm::data;

volume_data_loader_raw::volume_data_loader_raw()
    : volume_data_loader()
{
}

volume_data_loader_raw::~volume_data_loader_raw()
{
}

bool volume_data_loader_raw::open_file(const std::string& filename)
{
    math::vec3ui_t  dim;
    unsigned        num_chan;
    unsigned        bpc;

    using namespace boost::filesystem;
    path                file_path(filename, native);
    std::string         file_name       = file_path.leaf();
    std::string         file_extension  = extension(file_path);

    if (!parse_raw_file_name(file_name,
                             dim.x,
                             dim.y,
                             dim.z,
                             num_chan,
                             bpc)) {
        return (false);
    }

    if ((bpc % 8) != 0) {
        return (false);
    }
    bpc = bpc / 8;
    num_chan = 1;

    return (open_file(filename, dim, num_chan, bpc));
}

bool volume_data_loader_raw::open_file(const std::string& filename,
                                       const math::vec3ui_t& dim,
                                       unsigned num_chan,
                                       unsigned byte_per_chan)
{
    if (_file.is_open())
        _file.close();

    _file.open(filename.c_str(), std::ios::in | std::ios::binary);

    if (!_file) {
        return (false);
    }

    _vol_desc._data_dimensions       = dim;
    _vol_desc._data_byte_per_channel = byte_per_chan;
    _vol_desc._data_num_channels     = num_chan;

    // check if filesize checks out with given dimensions
    std::size_t len;

    _file.seekg (0, std::ios::end);
    len = _file.tellg();
    _file.seekg (0, std::ios::beg);

    if (len !=   _vol_desc._data_dimensions.x
               * _vol_desc._data_dimensions.y
               * _vol_desc._data_dimensions.z
               * _vol_desc._data_num_channels
               * _vol_desc._data_byte_per_channel) {
        return (false);
    }

    return (true);
}

bool volume_data_loader_raw::read_sub_volume(const math::vec<unsigned, 3>& offset,
                                             const math::vec<unsigned, 3>& dimensions,
                                             scm::data::regular_grid_data_3d<unsigned char>& target_data)
{
    if (!_file || !is_file_open()) {
        return (false);
    }

    // for now only ubyte and one channel data!
    if (_vol_desc._data_num_channels > 1 || _vol_desc._data_byte_per_channel > 1) {
        return (false);
    }

    if (   (offset.x + dimensions.x > _vol_desc._data_dimensions.x)
        || (offset.y + dimensions.y > _vol_desc._data_dimensions.y)
        || (offset.z + dimensions.z > _vol_desc._data_dimensions.z)) {
        return (false);
    }

    try {
        get_data_ptr(target_data).reset(new scm::data::regular_grid_data_3d<unsigned char>::value_type[dimensions.x * dimensions.y * dimensions.z]);
    }
    catch (std::bad_alloc&) {
        get_data_ptr(target_data).reset();
        return (false);
    }

    if (!read_sub_volume_data(offset, dimensions, get_data_ptr(target_data).get())) {
        get_data_ptr(target_data).reset();
        return (false);
    }

    set_dimensions(target_data, dimensions);
    target_data.update();

    return (true);
}
