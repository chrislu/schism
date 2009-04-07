
#ifndef SCM_IO_FILE_WIN_H_INCLUDED
#define SCM_IO_FILE_WIN_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <string>
#include <vector>

#include <scm/core/io/file_base.h>

#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {

namespace detail {

struct overlapped_ext;
struct io_result;

typedef scm::shared_ptr<overlapped_ext> request_ptr;

} // namespace detail

class __scm_export(core) file_win : public file_base
{
protected:
    typedef scm::shared_ptr<void>               handle;

public:
    file_win();
    file_win(const file_win& rhs);
    virtual ~file_win();

    file_win&                   operator=(const file_win& rhs);
    void                        swap(file_win& rhs);

    using file_base::swap;

    bool                        open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode                         = std::ios_base::in | std::ios_base::out,
                                     bool                     disable_system_cache              = true,
                                     scm::uint32              read_write_buffer_size            = detail::default_cache_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests  = detail::default_asynchronous_requests);
    bool                        is_open() const;
    void                        close();

    size_type                   read(char_type*const output_buffer,
                                     size_type       num_bytes_to_read);
    size_type                   write(const char_type*const input_buffer,
                                      size_type             num_bytes_to_write);

	size_type					set_end_of_file();

protected:
    size_type                   read_async(char_type*const output_buffer,
                                           size_type       num_bytes_to_read);
    bool                        read_async_request(const detail::request_ptr& req) const;

    size_type                   write_async(const char_type*const input_buffer,
                                            size_type             num_bytes_to_write);
    bool                        write_async_request(const detail::request_ptr& req) const;

    bool                        query_async_results(std::vector<detail::io_result>& res,
                                                    int query_max_results) const;

    void                        cancel_async_io() const;

    size_type                   actual_file_size() const;
    bool                        set_file_pointer(size_type new_pos);

    void                        reset_values();

protected:
    handle                      _file_handle;
    handle                      _completion_port;

}; // class file

} // namespace io
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#endif // SCM_IO_FILE_WIN_H_INCLUDED
