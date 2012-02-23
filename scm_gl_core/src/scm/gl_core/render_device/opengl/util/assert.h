
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_GL_ASSERT_H_INCLUDED
#define SCM_GL_CORE_GL_ASSERT_H_INCLUDED

#ifdef  gl_assert
#undef  gl_assert
#endif

#include <scm/gl_core/config.h>

#if SCM_GL_DEBUG

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {
namespace opengl {
class gl_core;
} // namespace opengl

__scm_export(gl_core) void _gl_assert(const opengl::gl_core& glcore,
                                      const char* message,
                                      const char* in_file,
                                      unsigned    at_line);

} // namespace gl
} // namespace scm

#define gl_assert(glcore, _message_) scm::gl::_gl_assert(glcore, TO_STR(_message_), __FILE__, __LINE__)

#else // SCM_GL_DEBUG

#define gl_assert(glcore, _message_) static_cast<void>(0)

#endif // SCM_GL_DEBUG


#endif // SCM_GL_CORE_GL_ASSERT_H_INCLUDED
