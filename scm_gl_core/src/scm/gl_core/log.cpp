
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

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
    gl_core_log.log_level(log::ll_output);
    return (gl_core_log.output());
}

log::out_stream
glerr()
{
#if SCM_GL_DEBUG
    gl_core_log.log_level(log::ll_debug);
    return (gl_core_log.debug());
#else
    gl_core_log.log_level(log::ll_error);
    return (gl_core_log.error());
#endif
}

} // namespace gl
} // namespace scm
