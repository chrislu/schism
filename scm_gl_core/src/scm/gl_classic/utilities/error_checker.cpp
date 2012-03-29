
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "error_checker.h"

namespace scm {
namespace gl_classic {

error_checker::error_checker() : _error(GL_NO_ERROR)
{
}

error_checker::~error_checker()
{
}

bool error_checker::ok()
{
    _error = glGetError();

    if (_error == GL_NO_ERROR) {
        return (true);
    }
    else {
        return (false);
    }
}

std::string error_checker::error_string()
{
    return (error_string(_error));
}

std::string error_checker::error_string(const GLenum error)
{
    std::string error_string;

    switch (error) {
        case GL_NO_ERROR:           error_string.assign("");                        break;
        case GL_INVALID_ENUM:       error_string.assign("GL_INVALID_ENUM");         break;
        case GL_INVALID_VALUE:      error_string.assign("GL_INVALID_VALUE");        break;
        case GL_INVALID_OPERATION:  error_string.assign("GL_INVALID_OPERATION");    break;
        case GL_STACK_OVERFLOW:     error_string.assign("GL_STACK_OVERFLOW");       break;
        case GL_STACK_UNDERFLOW:    error_string.assign("GL_STACK_UNDERFLOW");      break;
        case GL_OUT_OF_MEMORY:      error_string.assign("GL_OUT_OF_MEMORY");        break;
    };

    return (error_string);
}

} // namespace gl_classic
} // namespace scm
