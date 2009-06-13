
#ifndef ERROR_CHECKER_H_INCLUDED
#define ERROR_CHECKER_H_INCLUDED

#include <scm/gl.h>

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) error_checker
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

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // ERROR_CHECKER_H_INCLUDED
