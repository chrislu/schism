
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef CONFIG_H_INCLUDED
#define CONFIG_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#   define SCM_WIN_VER_2K          0x0500
#   define SCM_WIN_VER_XP          0x0501
#   define SCM_WIN_VER_XP_SP2      0x0502
#   define SCM_WIN_VER_VISTA       0x0600
#   define SCM_WIN_VER_DEFAULT     _WIN32_WINNT

// this is the target platform
#   define SCM_WIN_VER              SCM_WIN_VER_VISTA

#   undef   _WIN32_WINNT
#   define  _WIN32_WINNT            SCM_WIN_VER
#   undef   _WIN_VER
#   define  _WIN_VER                SCM_WIN_VER
#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#endif // CONFIG_H_INCLUDED
