
#ifndef SCM_GL_H_INCLUDED
#define SCM_GL_H_INCLUDED

#include <scm/core/platform/config.h>
#include <GL/glew.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

bool __scm_export(ogl) initialize();
bool __scm_export(ogl) shutdown();

} // namespace gl
} // namespace scm

#endif // SCM_GL_H_INCLUDED
