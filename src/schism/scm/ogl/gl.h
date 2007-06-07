
#ifndef SCM_GL_H_INCLUDED
#define SCM_GL_H_INCLUDED

//#ifdef _WIN32
//#include <windows.h>
//#endif
#include <GL/glew.h>

#include <string>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

bool __scm_export(ogl) initialize();
bool __scm_export(ogl) shutdown();

bool __scm_export(ogl) is_supported(const std::string&);

} // namespace gl
} // namespace scm

#endif // SCM_GL_H_INCLUDED
