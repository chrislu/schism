
#ifndef SCM_GL_CORE_LOG_H_INCLUDED
#define SCM_GL_CORE_LOG_H_INCLUDED

#include <boost/preprocessor/expand.hpp>

#include <scm/log.h>
#include <scm/gl_core/config.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

__scm_export(gl_core) log::out_stream  glout();
__scm_export(gl_core) log::out_stream  glerr();

} // namespace gl
} // namespace scm

#if SCM_GL_DEBUG
#define SCM_GL_DGB(X) scm::gl::glerr() << log::debug << BOOST_PP_EXPAND(X) << log::end
#else
#define SCM_GL_DGB(X) static_cast<void>(0)
#endif // SCM_GL_DEBUG

#endif // SCM_GL_CORE_LOG_H_INCLUDED
