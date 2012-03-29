
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_CORE_WINDOWS_H_INCLUDED
#define SCM_CORE_WINDOWS_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN 1
#   endif
#   include <windows.h>
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/config.h>

#endif // SCM_CORE_WINDOWS_H_INCLUDED
