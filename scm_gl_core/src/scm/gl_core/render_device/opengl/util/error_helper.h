
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_OPENGL_ERROR_HELPER_H_INCLUDED
#define SCM_GL_CORE_OPENGL_ERROR_HELPER_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

namespace opengl {

class gl_core;

} // namespace opengl

namespace util {

class __scm_export(gl_core) gl_error
{
    mutable unsigned        _error;
    const opengl::gl_core& _glcore;

public:
    gl_error(const opengl::gl_core& glcore);


    operator bool() const;

    bool                ok() const;
    std::string         error_string() const;
    static std::string  error_string(unsigned error);
    unsigned            to_object_state() const;
    static unsigned     to_object_state(unsigned error);
}; // class error_checker

} // namespace util
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_OPENGL_ERROR_HELPER_H_INCLUDED
