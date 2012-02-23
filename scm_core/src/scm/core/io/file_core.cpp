
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "file_core.h"

namespace scm {
namespace io {

file_core::file_core()
{
    reset_values();
}

file_core::~file_core()
{
}

// fixed functionality
file_core::offset_type
file_core::seek(offset_type                off,
                std::ios_base::seek_dir    way)
{
    offset_type  next_pos    = -1;

    switch (way) {
        case std::ios_base::beg:
            next_pos = off;
            break;
        case std::ios_base::cur:
            next_pos = _position + off;
            break;
        case std::ios_base::end:
            next_pos = _file_size + off;
            break;
    }

    assert(next_pos >= 0);

    _position = next_pos;

    return (_position);
}

file_core::size_type
file_core::optimal_buffer_size() const
{
    //if (_async_request_buffer_size && _volume_sector_size) {
    //    return (scm::math::max<scm::int32>(_volume_sector_size * 4, _rw_buffer_size / 16));
    //}
    //else {
        return (4096u);
    //}
}

file_core::size_type
file_core::size() const
{
    return (_file_size);
}

const std::string&
file_core::file_path() const
{
    return (_file_path);
}

scm::int32
file_core::volume_sector_size() const
{
    return (_volume_sector_size);
}

file_core::offset_type
file_core::vss_align_floor(const offset_type in_val) const
{
    assert(_volume_sector_size > 0);
    return((in_val / _volume_sector_size) * _volume_sector_size);
}

file_core::offset_type
file_core::vss_align_ceil(const offset_type in_val) const
{
    assert(_volume_sector_size > 0);
    return ( ((in_val / _volume_sector_size)
            + (in_val % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size);
}

void
file_core::reset_values()
{
    _position                   = 0;

    _file_path.clear();
    _file_size                  = 0;

    _open_mode                  = static_cast<std::ios_base::openmode>(0);

    _volume_sector_size         = 0;
    _async_requests             = 0;
    _async_request_buffer_size  = 0;
}

bool
file_core::async_io_mode() const
{
    return (_async_requests != 0 && _async_request_buffer_size != 0);
}

} // namepspace io
} // namepspace scm
