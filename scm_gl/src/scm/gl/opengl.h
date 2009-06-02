
#ifndef SCM_GL_H_INCLUDED
#define SCM_GL_H_INCLUDED

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#   include <scm/core/platform/windows.h>
#endif

#include <GL/glew.h>

// to use the direct state access extension define this token
//  - currently used in:
//    - texture objects
//    - program objects
#define SCM_GL_USE_DIRECT_STATE_ACCESS
//#undef SCM_GL_USE_DIRECT_STATE_ACCESS

namespace scm {
namespace gl {

} // namespace gl
} // namespace scm

#endif // SCM_GL_H_INCLUDED
