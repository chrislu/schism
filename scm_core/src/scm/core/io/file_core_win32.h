
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_IO_FILE_CORE_WIN32_H_INCLUDED
#define SCM_CORE_IO_FILE_CORE_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <vector>

#include <scm/core/memory.h>

#include <scm/core/io/file_core.h>

namespace scm {
namespace io {
namespace detail {

struct overlapped_ext;
struct io_result;

typedef scm::shared_ptr<overlapped_ext> request_ptr;

} // namespace detail

class file_core_win32 : public file_core
{
protected:
    typedef scm::shared_ptr<void>   handle;

public:
    file_core_win32();
    virtual ~file_core_win32();

    // file_core interface
    bool                        open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode,
                                     bool                     disable_system_cache,
                                     scm::uint32              read_write_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests);
    bool                        is_open() const;
    void                        close();

    size_type                   read(void*          output_buffer,
                                     offset_type    start_position,
                                     size_type      num_bytes_to_read);
    size_type                   write(const void*   input_buffer,
                                      offset_type   start_position,
                                      size_type     num_bytes_to_write);

    bool                        flush_buffers() const;

	offset_type                 set_end_of_file();
    // end file_core interface

private:
    size_type                   read_async(void*        output_buffer,
                                           offset_type  start_position,
                                           size_type    num_bytes_to_read);
    bool                        read_async_request(const detail::request_ptr& req) const;

    size_type                   write_async(const void* input_buffer,
                                            offset_type start_position,
                                            size_type   num_bytes_to_write);
    bool                        write_async_request(const detail::request_ptr& req) const;

    bool                        query_async_results(std::vector<detail::io_result>& res,
                                                    int query_max_results) const;

    void                        cancel_async_io() const;

    size_type                   actual_file_size() const;
    bool                        set_file_pointer(offset_type new_pos);

    void                        reset_values();

private:
    handle                      _file_handle;
    handle                      _completion_port;

}; // class file_core

} // namepspace io
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#endif // SCM_CORE_IO_FILE_CORE_WIN32_H_INCLUDED
