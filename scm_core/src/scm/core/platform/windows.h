
#ifndef SCM_CORE_WINDOWS_H_INCLUDED
#define SCM_CORE_WINDOWS_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#   define WIN32_LEAN_AND_MEAN
#   include <windows.h>
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/config.h>

#endif // SCM_CORE_WINDOWS_H_INCLUDED
