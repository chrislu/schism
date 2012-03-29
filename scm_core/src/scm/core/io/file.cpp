
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "file.h"

#include <cassert>

#include <scm/core/io/file_core.h>
#include <scm/core/io/file_core_win32.h>
#include <scm/core/io/file_core_linux.h>

namespace scm {
namespace io {

file::file()
{
#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS

    _file_core.reset(new file_core_win32);

#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX

    _file_core.reset(new file_core_linux);

#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif
}

file::~file()
{
}

// functionality depending on file_core
void
file::swap(file& rhs)
{
    _file_core.swap(rhs._file_core);
}

bool
file::open(const std::string&       file_path,
           std::ios_base::openmode  open_mode,
           bool                     disable_system_cache,
           scm::uint32              io_block_size,
           scm::uint32              async_io_requests)
{
    assert(_file_core);
    return _file_core->open(file_path,
                            open_mode,
                            disable_system_cache,
                            io_block_size,
                            async_io_requests);
}

bool
file::is_open() const
{
    assert(_file_core);
    return _file_core->is_open();
}

void
file::close()
{
    assert(_file_core);
    _file_core->close();
}

file::size_type
file::read(void*        output_buffer,
           offset_type  start_position,
           size_type    num_bytes_to_read)
{
    assert(_file_core);
    return _file_core->read(output_buffer, start_position, num_bytes_to_read);
}

file::size_type
file::write(const void* input_buffer,
            offset_type start_position,
            size_type   num_bytes_to_write)
{
    assert(_file_core);
    return _file_core->write(input_buffer, start_position, num_bytes_to_write);
}

bool
file::flush_buffers() const
{
    assert(_file_core);
    return _file_core->flush_buffers();
}

file::offset_type
file::set_end_of_file()
{
    assert(_file_core);
    return _file_core->set_end_of_file();
}

// fixed functionality
scm::int32
file::volume_sector_size() const
{
    assert(_file_core);
    return _file_core->volume_sector_size();
}

file::offset_type
file::vss_align_floor(const offset_type in_val) const
{
    assert(_file_core);
    return _file_core->vss_align_floor(in_val);
}

file::offset_type
file::vss_align_ceil(const offset_type in_val) const
{
    assert(_file_core);
    return _file_core->vss_align_ceil(in_val);
}

file::offset_type
file::seek(offset_type                off,
           std::ios_base::seek_dir    way)
{
    assert(_file_core);
    return _file_core->seek(off, way);
}

file::size_type
file::optimal_buffer_size() const
{
    assert(_file_core);
    return _file_core->optimal_buffer_size();
}

file::size_type
file::size() const
{
    assert(_file_core);
    return _file_core->size();
}

const std::string&
file::file_path() const
{
    assert(_file_core);
    return _file_core->file_path();
}

} // namespace io
} // namespace scm
