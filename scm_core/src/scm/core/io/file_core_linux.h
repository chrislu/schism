
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_IO_FILE_CORE_LINUX_H_INCLUDED
#define SCM_CORE_IO_FILE_CORE_LINUX_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <ios>
#include <vector>

#include <scm/core/memory.h>

#include <scm/core/io/file_core.h>

namespace scm {
namespace io {

class file_core_linux : public file_core
{
public:
    typedef shared_ptr<const int> handle;

public:
    file_core_linux();
    virtual ~file_core_linux();

    // file_core interface
    bool                        open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode,
                                     bool                     disable_system_cache,
                                     scm::uint32              read_write_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests);
    bool                        is_open() const;
    void                        close();

    size_type                   read(void*           output_buffer,
                                     offset_type     start_position,
                                     size_type       num_bytes_to_read);
    size_type                   write(const void*    input_buffer,
                                      offset_type    start_position,
                                      size_type      num_bytes_to_write);

    bool                        flush_buffers() const;

	offset_type			        set_end_of_file();
    // end file_core interface

private:
    size_type                   actual_file_size() const;
    bool                        set_file_pointer(offset_type new_pos);

    void                        reset_values();

private:
    handle                      _file_handle;

}; // class file_core_linux

} // namepspace io
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX

#endif // SCM_CORE_IO_FILE_CORE_LINUX_H_INCLUDED
