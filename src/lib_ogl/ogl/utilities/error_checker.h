
#ifndef ERROR_CHECKER_H_INCLUDED
#define ERROR_CHECKER_H_INCLUDED

#include <ogl/gl.h>

#include <string>

namespace gl
{
    class error_checker
    {
    public:
        error_checker();
        virtual ~error_checker();

        bool                ok();
        std::string         get_error_string(const GLenum /*error*/);
        std::string         get_error_string();

    protected:
        GLenum              _error;

    };
} // namespace gl

#endif // ERROR_CHECKER_H_INCLUDED



