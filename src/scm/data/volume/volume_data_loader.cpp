
#include "volume_data_loader.h"

namespace scm {
namespace data {

volume_data_loader::volume_data_loader()
  : _data_start_offset(0)
{
}

volume_data_loader::~volume_data_loader()
{
    if (is_file_open()){
        close_file();
    }
}

void volume_data_loader::close_file()
{
    _vol_desc = volume_descriptor();
    _file.close();
}

bool volume_data_loader::is_file_open() const
{
    return (_file.is_open());
}

const volume_descriptor& volume_data_loader::get_volume_descriptor() const
{
    return (_vol_desc);
}

bool volume_data_loader::read_volume_data(unsigned char*const buffer)
{
    if (!buffer || /*!_file ||*/ !is_file_open()) {
        return (false);
    }

    _file.seek(_data_start_offset, std::ios_base::beg);

    scm::int64 read_size =    _vol_desc._data_dimensions.x
                            * _vol_desc._data_dimensions.y
                            * _vol_desc._data_dimensions.z;

    if (_file.read((char*)buffer, read_size) != read_size) {
        return (false);
    }
    else {
        return (true);
    }
}

bool volume_data_loader::read_sub_volume_data(const scm::math::vec<unsigned, 3>& offset,
                                              const scm::math::vec3ui&           read_dimensions,
                                              const scm::math::vec3ui&           buffer_dimensions,
                                              unsigned char*const                buffer)
{
    if (!buffer || /*!_file ||*/ !is_file_open()) {
        return (false);
    }

    if (   (offset.x + read_dimensions.x > _vol_desc._data_dimensions.x)
        || (offset.y + read_dimensions.y > _vol_desc._data_dimensions.y)
        || (offset.z + read_dimensions.z > _vol_desc._data_dimensions.z)) {
        return (false);
    }
    
    unsigned offset_src;
    unsigned offset_dst;

#if 0
    for (unsigned int slice = 0; slice < read_dimensions.z; ++slice) {
        for (unsigned int line = 0; line < read_dimensions.y; ++line) {
            offset_src =  offset.x
                        + _vol_desc._data_dimensions.x * (offset.y + line)
                        + _vol_desc._data_dimensions.x * _vol_desc._data_dimensions.y * (offset.z + slice);
            offset_src *= _vol_desc._data_byte_per_channel * _vol_desc._data_num_channels;

            offset_dst =  buffer_dimensions.x * line
                        + buffer_dimensions.x * buffer_dimensions.y * slice;
            offset_dst *= _vol_desc._data_byte_per_channel * _vol_desc._data_num_channels;

            _file.seekg(offset_src + _data_start_offset, std::ios_base::beg);
            _file.read((char*)&buffer[offset_dst], read_dimensions.x);
        }
    }
#endif
    return (true);                                             
}

} // namespace data
} // namespace scm
