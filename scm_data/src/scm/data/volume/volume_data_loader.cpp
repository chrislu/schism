
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
    
    scm::int64 offset_src;
    scm::int64 offset_dst;

    bool success_reading = true;

    for (unsigned int slice = 0; slice < read_dimensions.z && success_reading; ++slice) {
        for (unsigned int line = 0; line < read_dimensions.y && success_reading; ++line) {
            offset_src =  static_cast<scm::int64>(offset.x)
                        + static_cast<scm::int64>(_vol_desc._data_dimensions.x) * static_cast<scm::int64>(offset.y + line)
                        + static_cast<scm::int64>(_vol_desc._data_dimensions.x) * static_cast<scm::int64>(_vol_desc._data_dimensions.y) * static_cast<scm::int64>(offset.z + slice);

            offset_src *= static_cast<scm::int64>(_vol_desc._data_byte_per_channel * _vol_desc._data_num_channels);

            offset_dst =  static_cast<scm::int64>(buffer_dimensions.x) * static_cast<scm::int64>(line)
                        + static_cast<scm::int64>(buffer_dimensions.x) * static_cast<scm::int64>(buffer_dimensions.y) * static_cast<scm::int64>(slice);
            offset_dst *= static_cast<scm::int64>(_vol_desc._data_byte_per_channel) * static_cast<scm::int64>(_vol_desc._data_num_channels);

            _file.seek(offset_src + static_cast<scm::int64>(_data_start_offset), std::ios_base::beg);
            if (_file.read((char*)&buffer[offset_dst], read_dimensions.x) != read_dimensions.x) {
                success_reading = false;
            }
        }
    }

    return (success_reading);                                             
}

} // namespace data
} // namespace scm
