
#ifndef SCM_IO_FILE_BASE_H_INCLUDED
#define SCM_IO_FILE_BASE_H_INCLUDED

#include <ios>

#include <scm/core/numeric_types.h>
#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {

namespace detail {
const scm::uint32   default_cache_buffer_size       = 32768u;
const scm::uint32   default_asynchronous_requests   = 8u;
} // namespace detail

class __scm_export(core) file_base
{
public:
    typedef char                char_type;
    typedef scm::int64          size_type;

public:
    file_base();
    file_base(const file_base& rhs);
    virtual ~file_base();

    //virtual file_base&          operator=(const file_base& rhs) = 0;
    virtual void                swap(file_base& rhs);

    virtual bool                open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode                         = std::ios_base::in | std::ios_base::out,
                                     bool                     disable_system_cache              = true,
                                     scm::uint32              read_write_buffer_size            = detail::default_cache_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests  = detail::default_asynchronous_requests) = 0;
    virtual bool                is_open() const = 0;
    virtual void                close() = 0;

    virtual size_type           read(char_type*const output_buffer,
                                     size_type       num_bytes_to_read) = 0;
    virtual size_type           write(const char_type*const input_buffer,
                                      size_type             num_bytes_to_write) = 0;

    virtual size_type           seek(size_type                  off,
                                     std::ios_base::seek_dir    way);

	virtual size_type			set_end_of_file() = 0;

    virtual size_type           optimal_buffer_size() const;

    size_type                   size() const;
    const std::string&          file_path() const;

    scm::int32                  volume_sector_size() const;
    size_type                   vss_align_floor(const size_type in_val) const;
    size_type                   vss_align_ceil(const size_type in_val) const;

protected:
    virtual void                reset_values();
    virtual bool                async_io_mode() const;

protected:
    size_type                   _position;

    std::string                 _file_path;
    size_type                   _file_size;

    std::ios_base::openmode     _open_mode;

    scm::int32                  _volume_sector_size;
    scm::int32                  _async_requests;
    scm::int32                  _async_request_buffer_size;

}; // class file_base

} // namespace io
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_IO_FILE_BASE_H_INCLUDED
