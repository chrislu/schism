
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "helper.h"

namespace scm {
namespace gl_classic {

/*static*/
GLenum fbo_status::_error = GL_NONE;

/*static*/
bool
fbo_status::ok()
{
    _error = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

    if (_error != GL_FRAMEBUFFER_COMPLETE_EXT) {
        return (false);
    }
    else {
        return (true);
    }
}

/*static*/
std::string
fbo_status::error_string()
{
    return (error_string(_error));
}

/*static*/
std::string
fbo_status::error_string(const GLenum error)
{
    std::string ret;

    switch (error) {
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:          ret.assign("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT");break;
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:  ret.assign("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT");break;
        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:          ret.assign("GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT");break;
        case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:             ret.assign("GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT");break;
        case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:         ret.assign("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT");break;
        case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:         ret.assign("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT");break;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:                    ret.assign("GL_FRAMEBUFFER_UNSUPPORTED_EXT");break;
    }

    return (ret);
}

} // namespace gl_classic
} // namespace scm
