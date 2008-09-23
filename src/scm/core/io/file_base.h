
#ifndef SCM_IO_FILE_BASE_H_INCLUDED
#define SCM_IO_FILE_BASE_H_INCLUDED

#include <ios>

#include <scm/core/int_types.h>
#include <scm/core/ptr_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {

namespace detail {
const scm::uint32   default_cache_buffer_size = 65536u;
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
                                     std::ios_base::openmode  open_mode = std::ios_base::in | std::ios_base::out,
                                     bool                     disable_system_cache = true,
                                     scm::uint32              read_write_buffer_size = detail::default_cache_buffer_size) = 0;
    virtual bool                is_open() const = 0;
    virtual void                close() = 0;

    virtual size_type           read(char_type* output_buffer,
                                     size_type  num_bytes_to_read) = 0;
    virtual size_type           write(const char_type* input_buffer,
                                      size_type        num_bytes_to_write) = 0;

    virtual size_type           seek(size_type                  off,
                                     std::ios_base::seek_dir    way);

    virtual size_type           optimal_buffer_size() const;

protected:
    virtual void                reset_values();

protected:
    size_type                   _position;

    std::string                 _file_path;
    size_type                   _file_size;

    std::ios_base::openmode     _open_mode;

    scm::shared_ptr<char_type>  _rw_buffer;
    scm::int32                  _rw_buffer_size;
    size_type                   _rw_buffered_start;
    size_type                   _rw_buffered_end;

    scm::int32                  _volume_sector_size;

}; // class file_base

} // namespace io
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_IO_FILE_BASE_H_INCLUDED
