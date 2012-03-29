
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_IO_FILE_H_INCLUDED
#define SCM_IO_FILE_H_INCLUDED

#include <ios>
#include <string>

#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/core/io/io_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {
namespace detail {

const scm::uint32   default_io_block_size           = 32768u;
const scm::uint32   default_asynchronous_requests   = 8u;

} // namespace detail

class file_core;

class __scm_export(core) file
{
public:
    typedef char                    char_type;
    typedef scm::io::size_type      size_type;
    typedef scm::io::offset_type    offset_type;

public:
    file();
    virtual ~file();

    void                        swap(file& rhs);

    // functionality depending on file_core
    bool                        open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode                 = std::ios_base::in | std::ios_base::out,
                                     bool                     disable_system_cache      = true,
                                     scm::uint32              io_block_size             = detail::default_io_block_size,
                                     scm::uint32              async_io_requests         = detail::default_asynchronous_requests);
    bool                        is_open() const;
    void                        close();
    size_type                   read(void*           output_buffer,
                                     offset_type     start_position,
                                     size_type       num_bytes_to_read);
    size_type                   write(const void*    input_buffer,
                                      offset_type    start_position,
                                      size_type      num_bytes_to_write);
    bool                        flush_buffers() const;
    offset_type                 seek(offset_type                off,
                                     std::ios_base::seek_dir    way);
    offset_type                 set_end_of_file();

    scm::int32                  volume_sector_size() const;
    offset_type                 vss_align_floor(const offset_type in_val) const;
    offset_type                 vss_align_ceil(const offset_type in_val) const;

    size_type                   optimal_buffer_size() const;

    size_type                   size() const;
    const std::string&          file_path() const;

private:
    scm::shared_ptr<file_core>  _file_core;

    // compiler generated copy constructor and assign operator work, they clone the core pointer

}; // class file

} // namespace io
} // namespace scm

namespace std {

inline void
swap(scm::io::file& lhs,
     scm::io::file& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_IO_FILE_H_INCLUDED
