
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
#undef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

#define SCM_GL_CORE_OPENGL_40 1
#undef SCM_GL_CORE_OPENGL_40

#define SCM_GL_CORE_OPENGL_41 1
#undef SCM_GL_CORE_OPENGL_41

// scm_gl_core internal ///////////////////////////////////////////////////////////////////////////
#define SCM_GL_CORE_OPENGL_VERSION_330   330
#define SCM_GL_CORE_OPENGL_VERSION_400   400
#define SCM_GL_CORE_OPENGL_VERSION_410   410

#define SCM_GL_CORE_MIN_OPENGL_VERSION   SCM_GL_CORE_OPENGL_VERSION_330
#define SCM_GL_CORE_BASE_OPENGL_VERSION  SCM_GL_CORE_MIN_OPENGL_VERSION

#if SCM_GL_CORE_OPENGL_40
#undef SCM_GL_CORE_BASE_OPENGL_VERSION
#define SCM_GL_CORE_BASE_OPENGL_VERSION SCM_GL_CORE_OPENGL_VERSION_400
#endif // SCM_GL_CORE_OPENGL_40

#if SCM_GL_CORE_OPENGL_41
#undef SCM_GL_CORE_BASE_OPENGL_VERSION
#define SCM_GL_CORE_BASE_OPENGL_VERSION SCM_GL_CORE_OPENGL_VERSION_410
#endif // SCM_GL_CORE_OPENGL_41

#endif // SCM_GL_CORE_OPENGL_CONFIG_H_INCLUDED
