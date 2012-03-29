
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_FBO_HELPER_H_INCLUDED
#define SCM_GL_FBO_HELPER_H_INCLUDED

#include <scm/gl_classic.h>

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) fbo_status
{
public:
    static bool         ok();
    static std::string  error_string();
    static std::string  error_string(const GLenum /*error*/);

protected:
    static GLenum       _error;

}; // class fbo_status

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>


#endif // SCM_GL_FBO_HELPER_H_INCLUDED
