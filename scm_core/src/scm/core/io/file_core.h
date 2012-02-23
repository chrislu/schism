
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_IO_FILE_CORE_H_INCLUDED
#define SCM_CORE_IO_FILE_CORE_H_INCLUDED

#include <boost/noncopyable.hpp>

#include <scm/core/io/file.h>

namespace scm {
namespace io {

class file_core : boost::noncopyable
{
public:
    typedef file::size_type     size_type;
    typedef file::offset_type   offset_type;
    typedef file::char_type     char_type;

public:
    file_core();
    virtual ~file_core();

    virtual bool                open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode,
                                     bool                     disable_system_cache,
                                     scm::uint32              read_write_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests) = 0;
    virtual bool                is_open() const = 0;
    virtual void                close() = 0;

    virtual size_type           read(void*           output_buffer,
                                     offset_type     start_position,
                                     size_type       num_bytes_to_read) = 0;
    virtual size_type           write(const void*    input_buffer,
                                      offset_type    start_position,
                                      size_type      num_bytes_to_write) = 0;

    virtual bool                flush_buffers() const = 0;

    virtual offset_type         set_end_of_file() = 0;

    // fixed functionality
    offset_type                 seek(offset_type                off,
                                     std::ios_base::seek_dir    way);
    size_type                   optimal_buffer_size() const;

    size_type                   size() const;
    const std::string&          file_path() const;

    scm::int32                  volume_sector_size() const;
    offset_type                 vss_align_floor(const offset_type in_val) const;
    offset_type                 vss_align_ceil(const offset_type in_val) const;

protected:
    virtual void                reset_values();
    virtual bool                async_io_mode() const;

protected:
    offset_type                 _position;

    std::string                 _file_path;
    size_type                   _file_size;

    std::ios_base::openmode     _open_mode;

    scm::int32                  _volume_sector_size;
    scm::int32                  _async_requests;
    scm::int32                  _async_request_buffer_size;

}; // class file_core

} // namepspace io
} // namepspace scm

#endif // SCM_CORE_IO_FILE_CORE_H_INCLUDED
