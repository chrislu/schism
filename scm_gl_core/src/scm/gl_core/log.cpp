
#include "log.h"

#include <scm/core/log/core.h>
#include <scm/gl_core/config.h>

namespace  {

scm::log::logger& gl_core_log(scm::logger("scm.gl_core"));

} // namespace 

namespace scm {
namespace gl {

log::out_stream
glout()
{
    return (gl_core_log.output());
}

log::out_stream
glerr()
{
#if SCM_GL_DEBUG
    return (gl_core_log.debug());
#else
    return (gl_core_log.error());
#endif
}

} // namespace gl
} // namespace scm
