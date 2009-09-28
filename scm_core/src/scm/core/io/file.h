
#ifndef SCM_IO_FILE_H_INCLUDED
#define SCM_IO_FILE_H_INCLUDED

#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#include <scm/core/io/file_win.h>
#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#error "atm unsupported platform"
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif

#include <boost/noncopyable.hpp>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace io {

namespace detail {
#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS

typedef file_win        file_impl;

#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#error "atm unsupported platform"
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif
} // namespace detail

class __scm_export(core) file : public detail::file_impl
{
public:
    file();
    file(const file& rhs);
    virtual ~file();

    file&           operator=(const file& rhs);

    using detail::file_impl::swap;
    using detail::file_impl::open;
    using detail::file_impl::is_open;
    using detail::file_impl::close;
    using detail::file_impl::read;
    using detail::file_impl::write;
    using detail::file_impl::seek;
    using detail::file_impl::size;
    using detail::file_impl::file_path;
    using detail::file_impl::set_end_of_file;
    using detail::file_impl::optimal_buffer_size;
}; // class file

#if 0

namespace detail {
const scm::uint32   default_cache_buffer_size       = 32768u;
const scm::uint32   default_asynchronous_requests   = 8u;
} // namespace detail

class file_core;

class __scm_export(core) file : boost::noncopyable
{
public:
    file();
    virtual ~file();

    // functionality depending on file_core
    void                        swap(file& rhs);

    bool                        open(const std::string&       file_path,
                                     std::ios_base::openmode  open_mode                         = std::ios_base::in | std::ios_base::out,
                                     bool                     disable_system_cache              = true,
                                     scm::uint32              read_write_buffer_size            = detail::default_cache_buffer_size,
                                     scm::uint32              read_write_asynchronous_requests  = detail::default_asynchronous_requests) = 0;
    bool                        is_open() const = 0;
    void                        close() = 0;
    size_type                   read(char_type*const output_buffer,
                                     size_type       num_bytes_to_read) = 0;
    size_type                   write(const char_type*const input_buffer,
                                      size_type             num_bytes_to_write) = 0;
	size_type			        set_end_of_file() = 0;

    // fixed functionality
    size_type                   seek(size_type                  off,
                                     std::ios_base::seek_dir    way);
    size_type                   optimal_buffer_size() const;

    size_type                   size() const;
    const std::string&          file_path() const;

    scm::int32                  volume_sector_size() const;
    size_type                   vss_align_floor(const size_type in_val) const;
    size_type                   vss_align_ceil(const size_type in_val) const;

private:
    size_type                   _position;

    std::string                 _file_path;
    size_type                   _file_size;

    std::ios_base::openmode     _open_mode;

    scm::int32                  _volume_sector_size;
    scm::int32                  _async_requests;
    scm::int32                  _async_request_buffer_size;

}; // class file

#endif

} // namespace io
} // namespace scm

namespace std {

inline void swap(scm::io::file& lhs,
                 scm::io::file& rhs)
{
    lhs.swap(rhs);
}

} // namespace std

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_IO_FILE_H_INCLUDED
