
#include "volume_data_loader_raw.h"

#include <exception>
#include <stdexcept>

#pragma warning (disable : 4561 4793)
#include <boost/spirit/core.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

#include <data_analysis/regular_grid_data_3d_write_accessor.h>

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

namespace gl
{
    volume_data_loader_raw::volume_data_loader_raw()
        : volume_data_loader()
    {
    }

    volume_data_loader_raw::~volume_data_loader_raw()
    {
    }

    bool volume_data_loader_raw::open_file(const std::string& filename)
    {
        using namespace boost::filesystem;
        path                file_path(filename, native);
        std::string         file_name       = file_path.leaf();
        std::string         file_extension  = extension(file_path);

            
        if (!parse_raw_file_name(file_name, _dimensions.x, _dimensions.y, _dimensions.z, _num_channels, _byte_per_channel)) {
            return (false);
        }

        if ((_byte_per_channel % 8) != 0) {
            return (false);
        }
        _byte_per_channel = _byte_per_channel / 8;
        _num_channels = 1;

        if (_file.is_open())
            _file.close();

        _file.open(filename.c_str(), std::ios::in | std::ios::binary);

        if (!_file) {
            return (false);
        }

        // check if filesize checks out with given dimensions
        int len;

        _file.seekg (0, std::ios::end);
        len = _file.tellg();
        _file.seekg (0, std::ios::beg);

        if (len !=   _dimensions.x
                   * _dimensions.y
                   * _dimensions.z
                   * _num_channels
                   * _byte_per_channel) {
            return (false);
        }

        return (true);
    }
    
    bool volume_data_loader_raw::read_sub_volume(const math::vec<unsigned, 3>& offset,
                                                 const math::vec<unsigned, 3>& dimensions,
                                                 scm::regular_grid_data_3d<unsigned char>& target_data)
    {
        if (!_file || !is_file_open()) {
            return (false);
        }

        // for now only ubyte and one channel data!
        if (_num_channels > 1 || _byte_per_channel > 1) {
            return (false);
        }

        if (   (offset.x + dimensions.x > _dimensions.x)
            || (offset.y + dimensions.y > _dimensions.y)
            || (offset.z + dimensions.z > _dimensions.z)) {
            return (false);
        }

        try {
            get_data_ptr(target_data).reset(new scm::regular_grid_data_3d<unsigned char>::value_type[dimensions.x * dimensions.y * dimensions.z]);
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

} // namespace gl
