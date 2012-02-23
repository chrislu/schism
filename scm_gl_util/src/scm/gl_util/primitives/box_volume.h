
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_BOX_VOLUME_H_INCLUDED
#define SCM_GL_UTIL_BOX_VOLUME_H_INCLUDED

#include <scm/core/math.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/primitives/box.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) box_volume_geometry : public box_geometry
{
public:
    box_volume_geometry(const render_device_ptr& in_device,
                 const math::vec3f&       in_min_vertex,
                 const math::vec3f&       in_max_vertex);
    virtual ~box_volume_geometry();

    using box_geometry::draw;

    void                update(const render_context_ptr& in_context,
                               const math::vec3f&        in_min_vertex,
                               const math::vec3f&        in_max_vertex);

}; // class box_volume_geometry

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_BOX_VOLUME_H_INCLUDED
