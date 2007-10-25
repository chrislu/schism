
#ifndef SCM_OGL_H_INCLUDED
#define SCM_OGL_H_INCLUDED

#include <scm/core/core_system_singleton.h>
#include <scm/ogl/system/opengl_system.h>
#include <scm/core/platform/platform.h>

#include <scm/ogl/gl.h>

namespace scm {

extern __scm_export(ogl) core::core_system_singleton<gl::opengl_system>::type  ogl;

} // namespace scm

#endif // SCM_OGL_H_INCLUDED
