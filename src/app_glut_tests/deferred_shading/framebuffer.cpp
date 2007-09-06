
#include "framebuffer.h"

#include <iostream>

#include <scm/ogl/gl.h>
#include <scm/ogl/utilities/error_checker.h>

namespace scm {

ds_framebuffer::ds_framebuffer(unsigned width,
                               unsigned height)
  : _id(0),
    _depth_id(0),
    _color_id(0),
    _normal_id(0)
{
    if (!init_textures(width, height)) {
        std::cout << "this is when RAII goes wrong..." << std::endl;
        return;
    }
    
    if (!init_fbo()) {
        std::cout << "this is when RAII goes wrong..." << std::endl;
        return;
    }
}

ds_framebuffer::~ds_framebuffer()
{
    cleanup();
}

bool ds_framebuffer::init_textures(unsigned width,
                                   unsigned height)
{
    scm::gl::error_checker error_check;

    // color buffer textures

    // color g-buffer
    glGenTextures(1, &_color_id);
    if (_color_id == 0) {
        std::cout << "unable to generate geometry fbo color renderbuffer texture" << std::endl;
        return (false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _color_id);

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    if (!error_check.ok()) {
        std::cout << "error creating geometry fbo color renderbuffer texture: ";
        std::cout << error_check.get_error_string() << std::endl;
        return (false);
    }
    else {
        std::cout << "successfully created geometry fbo color renderbuffer texture" << std::endl;
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    // normal g-buffer
    glGenTextures(1, &_normal_id);
    if (_normal_id == 0) {
        std::cout << "unable to generate geometry fbo normal renderbuffer texture" << std::endl;
        return (false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _normal_id);

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB,  width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    if (!error_check.ok()) {
        std::cout << "error creating geometry fbo normal renderbuffer texture: ";
        std::cout << error_check.get_error_string() << std::endl;
        return (false);
    }
    else {
        std::cout << "successfully created geometry fbo normal renderbuffer texture" << std::endl;
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);


    // depth g-buffer
    glGenTextures(1, &_depth_id);
    if (_depth_id == 0) {
        std::cout << "unable to generate geometry fbo depth renderbuffer texture" << std::endl;
        return (false);
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _depth_id);

    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH_COMPONENT24,  width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    if (!error_check.ok()) {
        std::cout << "error creating geometry fbo depth renderbuffer texture: ";
        std::cout << error_check.get_error_string() << std::endl;
        return (false);
    }
    else {
        std::cout << "successfully created geometry fbo depth renderbuffer texture" << std::endl;
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

    return (true);
}

bool ds_framebuffer::init_fbo()
{
    // create framebuffer object
    glGenFramebuffersEXT(1, &_id);
    if (_id == 0) {
        std::cout << "error generating fbo id" << std::endl;
        return (false);
    }
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _id);

    // attach depth texture
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_RECTANGLE_ARB, _depth_id, 0);

    // attach color textures
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, _color_id, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_TEXTURE_RECTANGLE_ARB, _normal_id, 0);

    unsigned fbo_status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    if (fbo_status != GL_FRAMEBUFFER_COMPLETE_EXT) {
        std::cout << "error creating fbo, framebufferstatus is not complete:" << std::endl;
        std::string error;

        switch (fbo_status) {
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:          error.assign("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:  error.assign("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:          error.assign("GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:             error.assign("GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:         error.assign("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT");break;
            case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:         error.assign("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT");break;
            case GL_FRAMEBUFFER_UNSUPPORTED_EXT:                    error.assign("GL_FRAMEBUFFER_UNSUPPORTED_EXT");break;
        }
        std::cout << "error: " << error << std::endl;
        return (false);
    }

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    return (true);
}

void ds_framebuffer::cleanup()
{
    glDeleteFramebuffersEXT(1, &_id);
    _id = 0;

    glDeleteTextures(1, &_depth_id);
    glDeleteTextures(1, &_color_id);
    glDeleteTextures(1, &_normal_id);
    _depth_id   = 0;
    _color_id   = 0;
    _normal_id  = 0;
}

} // namespace scm
