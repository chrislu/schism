
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
    _rw_buffer(rhs._rw_buffer),
    _rw_buffer_size(rhs._rw_buffer_size),
    _rw_buffered_start(rhs._rw_buffered_start),
    _rw_buffered_end(rhs._rw_buffered_end),
    _volume_sector_size(rhs._volume_sector_size)
{
}

file_base::~file_base()
{
}

void
file_base::swap(file_base& rhs)
{
    std::swap(_position,            rhs._position);
    std::swap(_file_path,           rhs._file_path);
    std::swap(_file_size,           rhs._file_size);
    std::swap(_open_mode,           rhs._open_mode);
    std::swap(_rw_buffer,           rhs._rw_buffer);
    std::swap(_rw_buffer_size,      rhs._rw_buffer_size);
    std::swap(_rw_buffered_start,   rhs._rw_buffered_start);
    std::swap(_rw_buffered_end,     rhs._rw_buffered_end);
    std::swap(_volume_sector_size,  rhs._volume_sector_size);
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

void
file_base::reset_values()
{
    _position               = 0;

    _file_path.clear();
    _file_size              = 0;

    _open_mode              = 0;

    _rw_buffer.reset();
    _rw_buffer_size         = 0;
    _rw_buffered_start      = 0;
    _rw_buffered_end        = 0;

    _volume_sector_size     = 0;
}

file_base::size_type
file_base::optimal_buffer_size() const
{
    if (_rw_buffer_size && _volume_sector_size) {
        return (scm::math::max<scm::int32>(_volume_sector_size * 4, _rw_buffer_size / 16));
    }
    else {
        return (4096u);
    }
}

} // namespace io
} // namespace scm
