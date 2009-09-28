
#include "file.h"

#include <cassert>

namespace scm {
namespace io {

#if !SCM_REBUILD_FILE

file::file()
  : detail::file_impl()
{
}

file::file(const file& rhs)
  : detail::file_impl(rhs)
{
}

file::~file()
{
}

file&
file::operator=(const file& rhs)
{
    file tmp(rhs);

    swap(tmp);

    return (*this);
}

#else

file::file()
{
}

file::~file()
{
}

// functionality depending on file_core
void
file::swap(file& rhs)
{
}

bool
file::open(const std::string&       file_path,
           std::ios_base::openmode  open_mode,
           bool                     disable_system_cache,
           scm::uint32              read_write_buffer_size,
           scm::uint32              read_write_asynchronous_requests)
{
    return (false);
}

bool
file::is_open() const
{
    return (false);
}

void
file::close()
{
}

file::size_type
file::read(char_type*const output_buffer,
           size_type       num_bytes_to_read)
{
    return (0);
}

file::size_type
file::write(const char_type*const input_buffer,
            size_type             num_bytes_to_write)
{
    return (0);
}

file::size_type
file::set_end_of_file()
{
    return (0);
}

// fixed functionality
file::size_type
file::seek(size_type                  off,
           std::ios_base::seek_dir    way)
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

file::size_type
file::optimal_buffer_size() const
{
    //if (_async_request_buffer_size && _volume_sector_size) {
    //    return (scm::math::max<scm::int32>(_volume_sector_size * 4, _rw_buffer_size / 16));
    //}
    //else {
        return (4096u);
    //}
}

file::size_type
file::size() const
{
    return (_file_size);
}

const std::string&
file::file_path() const
{
    return (_file_path);
}

scm::int32
file::volume_sector_size() const
{
    return (_volume_sector_size);
}

file::size_type
file::vss_align_floor(const size_type in_val) const
{
    return((in_val / _volume_sector_size) * _volume_sector_size);
}

file::size_type
file::vss_align_ceil(const size_type in_val) const
{
    return ( ((in_val / _volume_sector_size)
            + (in_val % _volume_sector_size > 0 ? 1 : 0)) * _volume_sector_size);
}

#endif

} // namespace io
} // namespace scm
