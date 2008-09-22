
#ifndef SCM_IO_FILE_H_INCLUDED
#define SCM_IO_FILE_H_INCLUDED

#include <ios>

#include <scm/core/int_types.h>
#include <scm/core/ptr_types.h>

namespace scm {
namespace io {

namespace detail {
const scm::uint32   default_cache_buffer_size = 65536u;
} // namespace detail

class file
{
public:
    typedef char                char_type;
    typedef scm::int64          size_type;

public:
    file();
    file(const file& rhs);
    virtual ~file();

    virtual file&               operator=(const file& rhs) = 0;
    virtual void                swap(file& rhs);

    virtual bool                open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode = std::ios_base::in | std::ios_base::out,
                                     bool                     disable_system_cache = true,
                                     scm::uint32              read_write_buffer_size = detail::default_cache_buffer_size) = 0;
    virtual bool                is_open() const = 0;
    virtual void                close() = 0;

    virtual size_type           read(char_type* s,          size_type n) = 0;
    virtual size_type           write(const char_type* s,   size_type n) = 0;

    virtual size_type           seek(size_type                  off,
                                     std::ios_base::seek_dir    way);

protected:
    size_type                   _position;

    std::string                 _file_path;
    size_type                   _file_size;

    std::ios_base::openmode     _open_mode;

    scm::shared_ptr<char_type>  _rw_buffer;
    scm::int32                  _rw_buffer_size;
    char_type                   _rw_buffered_start;
    char_type                   _rw_buffered_end;

}; // class file

} // namespace io
} // namespace scm

#endif // SCM_IO_FILE_H_INCLUDED
