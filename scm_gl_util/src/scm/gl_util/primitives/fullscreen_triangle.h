
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_FULLSCREEN_TRIANGLE_H_INCLUDED
#define SCM_GL_UTIL_FULLSCREEN_TRIANGLE_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/primitives/geometry.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) fullscreen_triangle : public geometry
{
public:
    fullscreen_triangle(const render_device_ptr& in_device);
    virtual ~fullscreen_triangle();

    void                draw(const render_context_ptr& in_context,
                             const draw_mode           in_draw_mode = MODE_SOLID) const;

protected:
    buffer_ptr          _vertices;
    vertex_array_ptr    _vertex_array;

}; // class fullscreen_triangle

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_FULLSCREEN_TRIANGLE_H_INCLUDED
