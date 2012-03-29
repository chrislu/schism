
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef TEXTURE_2D_RECT_H_INCLUDED
#define TEXTURE_2D_RECT_H_INCLUDED

#include <scm/gl_classic/textures/texture_2d.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) texture_2d_rect : public texture_2d
{
public:
    texture_2d_rect();
    virtual ~texture_2d_rect();

protected:

private:

}; // class texture_2d_rect

} // namespace scm
} // namespace gl_classic

#include <scm/core/utilities/platform_warning_enable.h>

#endif // TEXTURE_2D_RECT_H_INCLUDED
