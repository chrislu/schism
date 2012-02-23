
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_GEOMETRY_H_INCLUDED
#define SCM_GL_UTIL_GEOMETRY_H_INCLUDED

#include <scm/gl_core/gl_core_fwd.h>

#include <scm/gl_util/primitives/primitives_fwd.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) geometry
{
public:
    enum draw_mode {
        MODE_SOLID      = 0x00,
        MODE_WIRE_FRAME,
    };
public:
    geometry(const render_device_ptr& in_device);
    virtual ~geometry();

    virtual void        draw(const render_context_ptr& in_context,
                             const draw_mode           in_draw_mode = MODE_SOLID) const = 0;

protected:

private:

}; // class geometry

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_GEOMETRY_H_INCLUDED
