
#include "file.h"

#include <algorithm>

namespace scm {
namespace io {

file::file()
  : _position(0),
    _file_size(0),
    _open_mode(0),
    _rw_buffer_size(0),
    _rw_buffered_start(0),
    _rw_buffered_end(0)
{
}

file::file(const file& rhs)
  : _position(rhs._position),
    _file_path(rhs._file_path),
    _file_size(rhs._file_size),
    _open_mode(rhs._open_mode),
    _rw_buffer(rhs._rw_buffer),
    _rw_buffer_size(rhs._rw_buffer_size),
    _rw_buffered_start(rhs._rw_buffered_start),
    _rw_buffered_end(rhs._rw_buffered_end)
{
}

file::~file()
{
}

void
file::swap(file& rhs)
{
    std::swap(_position,            rhs._position);
    std::swap(_file_path,           rhs._file_path);
    std::swap(_file_size,           rhs._file_size);
    std::swap(_open_mode,           rhs._open_mode);
    std::swap(_rw_buffer,           rhs._rw_buffer);
    std::swap(_rw_buffer_size,      rhs._rw_buffer_size);
    std::swap(_rw_buffered_start,   rhs._rw_buffered_start);
    std::swap(_rw_buffered_end,     rhs._rw_buffered_end);
}

file::size_type
file::seek(file::size_type          off,
           std::ios_base::seek_dir  way)
{
    return (0);
}

} // namespace io
} // namespace scm
