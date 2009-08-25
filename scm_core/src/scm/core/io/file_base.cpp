
#include "file_base.h"

#include <algorithm>
#include <cassert>

#include <scm/core/math/math.h>

namespace scm {
namespace io {

file_base::file_base()
{
    reset_values();
}

file_base::file_base(const file_base& rhs)
  : _position(rhs._position),
    _file_path(rhs._file_path),
    _file_size(rhs._file_size),
    _open_mode(rhs._open_mode),
    _volume_sector_size(rhs._volume_sector_size),
    _async_requests(rhs._async_requests),
    _async_request_buffer_size(rhs._async_request_buffer_size)
{
}

file_base::~file_base()
{
}

void
file_base::swap(file_base& rhs)
{
    std::swap(_position,                    rhs._position);
    std::swap(_file_path,                   rhs._file_path);
    std::swap(_file_size,                   rhs._file_size);
    std::swap(_open_mode,                   rhs._open_mode);
    std::swap(_volume_sector_size,          rhs._volume_sector_size);
    std::swap(_async_requests,              rhs._async_requests);
    std::swap(_async_request_buffer_size,   rhs._async_request_buffer_size);
}

file_base::size_type
file_base::seek(file_base::size_type    off,
                std::ios_base::seek_dir way)
{
    size_type       next_pos    = -1;

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

file_base::size_type
file_base::size() const
{
    return (_file_size);
}

const std::string&
file_base::file_path() const
{
    return (_file_path);
}

scm::int32
file_base::volume_sector_size() const
{
    return (_volume_sector_size);
}

file_base::size_type
file_base::vss_align_floor(const file_base::size_type in_val) const
{
    return((in_val / _volume_sector_size) * _volume_sector_size);
}

file_base::size_type
file_base::vss_align_ceil(const file_base::size_type in_val) const
{
    return ( ((in_val / _volume_sector_size)
            + (in_val % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size);
}

void
file_base::reset_values()
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
file_base::async_io_mode() const
{
    return (_async_requests != 0 && _async_request_buffer_size != 0);
}

file_base::size_type
file_base::optimal_buffer_size() const
{
    //if (_async_request_buffer_size && _volume_sector_size) {
    //    return (scm::math::max<scm::int32>(_volume_sector_size * 4, _rw_buffer_size / 16));
    //}
    //else {
        return (4096u);
    //}
}

} // namespace io
} // namespace scm
