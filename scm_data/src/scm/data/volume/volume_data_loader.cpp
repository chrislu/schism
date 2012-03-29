
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

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
    _vol_desc = volume_file_descriptor();
    _file.close();
}

bool volume_data_loader::is_file_open() const
{
    return (_file.is_open());
}

const volume_file_descriptor& volume_data_loader::get_volume_descriptor() const
{
    return (_vol_desc);
}

bool volume_data_loader::read_volume_data(unsigned char*const buffer)
{
    if (!buffer || /*!_file ||*/ !is_file_open()) {
        return (false);
    }

    //_file.seek(_data_start_offset, std::ios_base::beg);

    scm::int64 read_size =    static_cast<scm::int64>(_vol_desc._data_dimensions.x)
                            * static_cast<scm::int64>(_vol_desc._data_dimensions.y)
                            * static_cast<scm::int64>(_vol_desc._data_dimensions.z)
                            * static_cast<scm::int64>(_vol_desc._data_byte_per_channel)
                            * static_cast<scm::int64>(_vol_desc._data_num_channels);

    if (_file.read((char*)buffer, _data_start_offset, read_size) != read_size) {
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

    const scm::int64                    data_value_size = static_cast<scm::int64>(_vol_desc._data_byte_per_channel) * static_cast<scm::int64>(_vol_desc._data_num_channels);
    const scm::math::vec<scm::int64, 3> offset64(offset);
    const scm::math::vec<scm::int64, 3> dimensions64(_vol_desc._data_dimensions);
    const scm::math::vec<scm::int64, 3> buf_dimensions64(buffer_dimensions);

    bool success_reading = true;

    for (unsigned int slice = 0; slice < read_dimensions.z && success_reading; ++slice) {
        for (unsigned int line = 0; line < read_dimensions.y && success_reading; ++line) {
            offset_src =  offset64.x
                        + dimensions64.x * (offset64.y + line)
                        + dimensions64.x * dimensions64.y * (offset64.z + slice);
            offset_src *= data_value_size;

            offset_dst =  buf_dimensions64.x * line
                        + buf_dimensions64.x * buf_dimensions64.y * slice;
            offset_dst *= data_value_size;

            //_file.seek(static_cast<scm::int64>(_data_start_offset) + offset_src, std::ios_base::beg);
            scm::int64 read_off  = _data_start_offset + offset_src;
            scm::int64 read_size = data_value_size * read_dimensions.x;

            if (_file.read((char*)&buffer[offset_dst], read_off, read_size) != read_size) {
                success_reading = false;
            }
        }
    }

    return (success_reading);                                             
}

} // namespace data
} // namespace scm
