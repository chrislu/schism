
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

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

#define SCM_GL_LOG_ONCE(L, X)                                                       \
    static bool scm_gl_log_once_done = false;                                       \
    if (!scm_gl_log_once_done) {                                                    \
        scm::gl::glout() << BOOST_PP_EXPAND(L) << BOOST_PP_EXPAND(X) << log::end;   \
    }

#endif // SCM_GL_CORE_LOG_H_INCLUDED
