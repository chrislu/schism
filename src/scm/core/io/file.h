
#ifndef SCM_IO_FILE_H_INCLUDED
#define SCM_IO_FILE_H_INCLUDED

#if    SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#include <scm/core/io/file_win.h>
#elif  SCM_PLATFORM == SCM_PLATFORM_LINUX
#error "atm unsupported platform"
#elif  SCM_PLATFORM == SCM_PLATFORM_APPLE
#error "atm unsupported platform"
#endif


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

class __scm_export(core) file : private detail::file_impl
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
    using detail::file_impl::optimal_buffer_size;
}; // class file

} // namespace io
} // namespace scm

#include <scm/core/utilities/platform_warning_disable.h>

#endif // SCM_IO_FILE_H_INCLUDED
