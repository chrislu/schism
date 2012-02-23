
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_OPENGL_CONFIG_H_INCLUDED
#define SCM_GL_CORE_OPENGL_CONFIG_H_INCLUDED

#include <scm/core/platform/platform.h>
// gl debugging
#if SCM_DEBUG
#   define SCM_GL_DEBUG 1
#else
#   define SCM_GL_DEBUG 0
#endif

// to use the direct state access extension define this token
//  - currently used in:
//    - texture objects
//    - program objects
#define SCM_GL_CORE_USE_DIRECT_STATE_ACCESS 1
//#undef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS


#endif // SCM_GL_CORE_OPENGL_CONFIG_H_INCLUDED
