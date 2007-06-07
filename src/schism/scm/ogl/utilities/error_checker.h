
#ifndef ERROR_CHECKER_H_INCLUDED
#define ERROR_CHECKER_H_INCLUDED

#include <scm/ogl/gl.h>

#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) error_checker
{
public:
    error_checker();
    virtual ~error_checker();

    bool                ok();
    std::string         get_error_string(const GLenum /*error*/);
    std::string         get_error_string();

protected:
    GLenum              _error;

}; // class error_checker

} // namespace gl
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // ERROR_CHECKER_H_INCLUDED
