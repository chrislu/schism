
#ifndef SCM_GL_CORE_OPENGL_ERROR_HELPER_H_INCLUDED
#define SCM_GL_CORE_OPENGL_ERROR_HELPER_H_INCLUDED

#include <string>

namespace scm {
namespace gl {

namespace opengl {

class gl3_core;

} // namespace opengl

namespace util {

class gl_error
{
    mutable unsigned        _error;
    const opengl::gl3_core& _glcore;

public:
    gl_error(const opengl::gl3_core& glcore);


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

#endif // SCM_GL_CORE_OPENGL_ERROR_HELPER_H_INCLUDED
