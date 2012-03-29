
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TEXTURE_2D_H_INCLUDED
#define TEXTURE_2D_H_INCLUDED

#include <scm/gl_classic/textures/texture.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) texture_2d : public texture
{
public:
    texture_2d();
    virtual ~texture_2d();

    unsigned        width() const       { return (_width); }
    unsigned        height() const      { return (_height); }

    bool            image_data(GLint     mip_level,
                               GLint     internal_format,
                               GLsizei   width,
                               GLsizei   height,
                               GLenum    format,
                               GLenum    type,
                               const GLvoid *data);

    bool            image_sub_data(GLint     mip_level,
                                   GLint     off_x,
                                   GLint     off_y,
                                   GLsizei   width,
                                   GLsizei   height,
                                   GLenum    format,
                                   GLenum    type,
                                   const GLvoid *data);

protected:
    texture_2d(const GLenum target, const GLenum binding);

    unsigned        _width;
    unsigned        _height;

private:

}; // class texture_2d

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TEXTURE_2D_H_INCLUDED
