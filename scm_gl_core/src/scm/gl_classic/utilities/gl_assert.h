
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_GL_ASSERT_H_INCLUDED
#define SCM_GL_GL_ASSERT_H_INCLUDED

#undef  gl_assert_error

#if SCM_DEBUG

#define gl_assert_error(_message_) static_cast<void>(0)

#else // SCM_DEBUG

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl_classic {

__scm_export(gl_core) void _gl_assert_error(const char* message,
                                        const char* in_file,
                                        unsigned    at_line);
} // namespace gl_classic
} // namespace scm

#define gl_assert_error(_message_) scm::gl_classic::_gl_assert_error(TO_STR(_message_), __FILE__, __LINE__)

#endif // SCM_DEBUG

#endif // SCM_GL_GL_ASSERT_H_INCLUDED
