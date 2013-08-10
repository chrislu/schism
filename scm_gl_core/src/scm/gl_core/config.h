
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_OPENGL_CONFIG_H_INCLUDED
#define SCM_GL_CORE_OPENGL_CONFIG_H_INCLUDED

#include <scm/core/platform/platform.h>
#include <scm/gl_core/config_defines.h>

// gl debugging
#if SCM_DEBUG
#   define SCM_GL_DEBUG 1
#else
#   define SCM_GL_DEBUG 0
#endif

#define SCM_GL_CORE_OPENGL_PROFILE          SCM_GL_CORE_OPENGL_440_PROFILE

// to use the direct state access extension define this token
//  - currently used in:
//    - texture objects
//    - program objects
//    - buffer objects
//    - vertex array objects
#define SCM_GL_CORE_USE_DIRECT_STATE_ACCESS 1
//#undef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

#define SCM_GL_CORE_WORKAROUND_AMD 1
#undef SCM_GL_CORE_WORKAROUND_AMD

// scm_gl_core internal ///////////////////////////////////////////////////////////////////////////
// helper macros //////////////////////////////////////////////////////////////////////////////////
#define SCM_GL_CORE_OPENGL_TYPE         ((SCM_GL_CORE_OPENGL_PROFILE) & SCM_GL_CORE_OPENGL_PLATFORM_MASK)
#define SCM_GL_CORE_OPENGL_VERSION      ((SCM_GL_CORE_OPENGL_PROFILE) & SCM_GL_CORE_OPENGL_VERSION_MASK)

#if SCM_GL_CORE_OPENGL_TYPE == SCM_GL_CORE_OPENGL_CORE_TYPE
#   define SCM_GL_CORE_OPENGL_CORE_VERSION      (SCM_GL_CORE_OPENGL_VERSION)
#else
#   define SCM_GL_CORE_OPENGL_CORE_VERSION      0u
#endif

#if SCM_GL_CORE_OPENGL_TYPE == SCM_GL_CORE_OPENGL_ES2_TYPE
#   define SCM_GL_CORE_OPENGL_ES2_VERSION       (SCM_GL_CORE_OPENGL_VERSION)
#else
#   define SCM_GL_CORE_OPENGL_ES2_VERSION       0u
#endif

// extension macros ///////////////////////////////////////////////////////////////////////////////
#if SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
#   define SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS 1
#else
#   define SCM_GL_CORE_USE_EXT_DIRECT_STATE_ACCESS 0
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

#if SCM_GL_CORE_WORKAROUND_AMD
#   define SCM_GL_CORE_USE_WORKAROUND_AMD 1
#else
#   define SCM_GL_CORE_USE_WORKAROUND_AMD 0
#endif // SCM_GL_CORE_WORKAROUND_AMD

// end scm_gl_core internal ///////////////////////////////////////////////////////////////////////

#endif // SCM_GL_CORE_OPENGL_CONFIG_H_INCLUDED
