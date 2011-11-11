
#ifndef SCM_GL_CORE_CL_UTIL_H_INCLUDED
#define SCM_GL_CORE_CL_UTIL_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace cl {
namespace util {

std::string
__scm_export(gl_core) cl_error_string(int error_code);

} // namespace util
} // namespace cl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_CL_UTIL_H_INCLUDED
