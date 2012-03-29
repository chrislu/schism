
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef ERROR_CHECKER_H_INCLUDED
#define ERROR_CHECKER_H_INCLUDED

#include <scm/gl_classic.h>

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) error_checker
{
public:
    error_checker();
    virtual ~error_checker();

    bool                ok();
    static std::string  error_string(const GLenum /*error*/);
    std::string         error_string();

protected:
    GLenum              _error;

}; // class error_checker

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // ERROR_CHECKER_H_INCLUDED
