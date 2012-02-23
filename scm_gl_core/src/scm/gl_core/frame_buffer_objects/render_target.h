
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_RENDER_TARGET_H_INCLUDED
#define SCM_GL_CORE_RENDER_TARGET_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/frame_buffer_objects/frame_buffer_objects_fwd.h>
#include <scm/gl_core/render_device/render_device_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) render_target
{
public:
    virtual ~render_target();

    virtual data_format     format() const = 0;
    virtual math::vec2ui    dimensions() const = 0;
    virtual unsigned        array_layers() const = 0;
    virtual unsigned        mip_map_layers() const = 0;
    virtual unsigned        samples() const = 0;

protected:
    render_target();

    virtual unsigned        object_id() const = 0;
    virtual unsigned        object_target() const = 0;

private:

    friend class render_device;
    friend class render_context;
    friend class frame_buffer;
}; // class render_target

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_RENDER_TARGET_H_INCLUDED
