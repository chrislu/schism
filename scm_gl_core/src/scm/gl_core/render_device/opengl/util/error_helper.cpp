
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "error_helper.h"

#include <scm/gl_core/object_state.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>

namespace scm {
namespace gl {
namespace util {

gl_error::gl_error(const opengl::gl_core& glcore)
  : _glcore(glcore)
{
}

std::string
gl_error::error_string() const
{
    return (error_string(_error));
}

std::string
gl_error::error_string(unsigned error)
{
    std::string error_string;

    switch (error) {
        case GL_NO_ERROR:           error_string.assign("");                        break;
        case GL_INVALID_ENUM:       error_string.assign("GL_INVALID_ENUM");         break;
        case GL_INVALID_VALUE:      error_string.assign("GL_INVALID_VALUE");        break;
        case GL_INVALID_OPERATION:  error_string.assign("GL_INVALID_OPERATION");    break;
        //case GL_STACK_OVERFLOW:     error_string.assign("GL_STACK_OVERFLOW");       break;
        //case GL_STACK_UNDERFLOW:    error_string.assign("GL_STACK_UNDERFLOW");      break;
        case GL_OUT_OF_MEMORY:      error_string.assign("GL_OUT_OF_MEMORY");        break;
        default:                    error_string.assign("unknown error");           break;
    };

    return (error_string);
}

unsigned
gl_error::to_object_state() const
{
    return (to_object_state(_error));
}

unsigned
gl_error::to_object_state(unsigned error)
{
    unsigned out_state = object_state::OS_ERROR_UNKNOWN;

    switch (error) {
        case GL_NO_ERROR:               out_state = object_state::OS_OK;                        break;
        case GL_INVALID_ENUM:           out_state = object_state::OS_ERROR_INVALID_ENUM;        break;
        case GL_INVALID_VALUE:          out_state = object_state::OS_ERROR_INVALID_VALUE;       break;
        case GL_INVALID_OPERATION:      out_state = object_state::OS_ERROR_INVALID_OPERATION;   break;
        case GL_OUT_OF_MEMORY:          out_state = object_state::OS_ERROR_OUT_OF_MEMORY;       break;
    };
    return (out_state);
}

gl_error::operator bool() const
{
    return (!ok());
}

bool  
gl_error::ok() const
{
    _error = _glcore.glGetError();

    if (_error == GL_NO_ERROR) {
        return (true);
    }
    else {
        return (false);
    }
}

} // namespace util
} // namespace gl
} // namespace scm
