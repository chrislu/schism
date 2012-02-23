
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WIN32_ERROR_WIN32_H_INCLUDED
#define SCM_GL_CORE_WM_WIN32_ERROR_WIN32_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <string>

#include <boost/lexical_cast.hpp>

namespace scm {
namespace gl {
namespace wm {
namespace util {

inline
std::string win32_error_message()
{
    char* error_msg;

    if (0 == FormatMessage(    FORMAT_MESSAGE_IGNORE_INSERTS
                             | FORMAT_MESSAGE_FROM_SYSTEM
                             | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                             0,
                             GetLastError(),
                             MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                             (LPTSTR)&error_msg,
                             0,
                             0)) {
        return (std::string("FormatMessage failed (" + boost::lexical_cast<std::string>(GetLastError())));
    }

    std::string msg(error_msg);
    LocalFree(error_msg);

    return (msg);
}

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_CORE_WM_WIN32_ERROR_WIN32_H_INCLUDED
