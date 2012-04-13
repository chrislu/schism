
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "box_volume.h"

#include <cassert>
#include <exception>
#include <stdexcept>

#include <boost/assign/list_of.hpp>

#include <scm/core/memory.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/buffer_objects/scoped_buffer_map.h>
#include <scm/gl_core/buffer_objects/vertex_array.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>

namespace {
    struct vertex {
        scm::math::vec3f pos;
        scm::math::vec3f nrm;
        scm::math::vec3f tex;
    };
} // namespace

namespace scm {
namespace gl {

box_volume_geometry::box_volume_geometry(const render_device_ptr& in_device,
                                         const math::vec3f& in_min_vertex,
                                         const math::vec3f& in_max_vertex)
  : box_geometry(in_device, in_min_vertex, in_max_vertex)
{
    using namespace scm::math;

    int num_vertices            = 6 * 4; // 4 vertices per face

    _vertices = in_device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, num_vertices * sizeof(vertex), 0);

    render_context_ptr ctx = in_device->main_context();
    vertex* data = static_cast<vertex*>(ctx->map_buffer(_vertices, ACCESS_WRITE_INVALIDATE_BUFFER));

    const vec3f& o = in_min_vertex;
    const vec3f& l = in_max_vertex;

    // front face
    int v = 0;
    data[v].pos  = vec3f(o.x, o.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec3f(0.0f, 0.0f, 1.0f);++v;
    data[v].pos  = vec3f(l.x, o.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec3f(1.0f, 0.0f, 1.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec3f(1.0f, 1.0f, 1.0f);++v;
    data[v].pos  = vec3f(o.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec3f(0.0f, 1.0f, 1.0f);++v;
    // right face
    data[v].pos  = vec3f(l.x, o.y, l.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec3f(1.0f, 0.0f, 1.0f);++v;
    data[v].pos  = vec3f(l.x, o.y, o.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec3f(1.0f, 0.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, o.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec3f(1.0f, 1.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, l.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec3f(1.0f, 1.0f, 1.0f);++v;
    // back face
    data[v].pos  = vec3f(l.x, o.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec3f(1.0f, 0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(o.x, o.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec3f(0.0f, 0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(o.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec3f(0.0f, 1.0f, 0.0f); ++v;
    data[v].pos  = vec3f(l.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec3f(1.0f, 1.0f, 0.0f); ++v;
    // left face
    data[v].pos  = vec3f(o.x, o.y, o.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec3f(0.0f, 0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(o.x, o.y, l.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec3f(0.0f, 0.0f, 1.0f); ++v;
    data[v].pos  = vec3f(o.x, l.y, l.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec3f(0.0f, 1.0f, 1.0f); ++v;
    data[v].pos  = vec3f(o.x, l.y, o.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec3f(0.0f, 1.0f, 0.0f); ++v;
    // top face
    data[v].pos  = vec3f(o.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec3f(0.0f, 1.0f, 0.0f);++v;
    data[v].pos  = vec3f(o.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec3f(0.0f, 1.0f, 1.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec3f(1.0f, 1.0f, 1.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec3f(1.0f, 1.0f, 0.0f);++v;
    // bottom face
    data[v].pos  = vec3f(o.x, o.y, o.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec3f(0.0f, 0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(l.x, o.y, o.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec3f(1.0f, 0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(l.x, o.y, l.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec3f(1.0f, 0.0f, 1.0f); ++v;
    data[v].pos  = vec3f(o.x, o.y, l.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec3f(0.0f, 0.0f, 1.0f); ++v;

    ctx->unmap_buffer(_vertices);

    assert(v == (6 * 4));

    using namespace scm::gl;
    using boost::assign::list_of;

    _vertex_array = in_device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F, sizeof(vertex))
                                                                (0, 1, TYPE_VEC3F, sizeof(vertex))
                                                                (0, 2, TYPE_VEC3F, sizeof(vertex)),
                                                   list_of(_vertices));

}

box_volume_geometry::~box_volume_geometry()
{
}

void
box_volume_geometry::update(const render_context_ptr& in_context,
                     const math::vec3f& in_min_vertex,
                     const math::vec3f& in_max_vertex)
{
    using namespace scm::math;

    scoped_buffer_map vb_map(in_context, _vertices, ACCESS_WRITE_INVALIDATE_BUFFER);

    if (!vb_map) {
        glerr() << log::error
                << "box_volume_geometry::update(): unable to map vertex buffer." << log::end;
        return;
    }

    vertex*const    data = reinterpret_cast<vertex*const>(vb_map.data_ptr());

    const vec3f& o = in_min_vertex;
    const vec3f& l = in_max_vertex;

    // front face
    int v = 0;
    data[v].pos  = vec3f(o.x, o.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec2f(0.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, o.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec2f(1.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec2f(1.0f, 1.0f);++v;
    data[v].pos  = vec3f(o.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 0.0f, 1.0f);  data[v].tex = vec2f(0.0f, 1.0f);++v;
    // right face
    data[v].pos  = vec3f(l.x, o.y, l.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec2f(0.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, o.y, o.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec2f(1.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, o.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec2f(1.0f, 1.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, l.z); data[v].nrm = vec3f(1.0f, 0.0f, 0.0f);  data[v].tex = vec2f(0.0f, 1.0f);++v;
    // back face
    data[v].pos  = vec3f(l.x, o.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec2f(0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(o.x, o.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec2f(1.0f, 0.0f); ++v;
    data[v].pos  = vec3f(o.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec2f(1.0f, 1.0f); ++v;
    data[v].pos  = vec3f(l.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 0.0f, -1.0f); data[v].tex = vec2f(0.0f, 1.0f); ++v;
    // left face
    data[v].pos  = vec3f(o.x, o.y, o.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec2f(0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(o.x, o.y, l.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec2f(1.0f, 0.0f); ++v;
    data[v].pos  = vec3f(o.x, l.y, l.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec2f(1.0f, 1.0f); ++v;
    data[v].pos  = vec3f(o.x, l.y, o.z); data[v].nrm = vec3f(-1.0f, 0.0f, 0.0f); data[v].tex = vec2f(0.0f, 1.0f); ++v;
    // top face
    data[v].pos  = vec3f(o.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec2f(0.0f, 1.0f);++v;
    data[v].pos  = vec3f(o.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec2f(0.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, l.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec2f(1.0f, 0.0f);++v;
    data[v].pos  = vec3f(l.x, l.y, o.z); data[v].nrm = vec3f(0.0f, 1.0f, 0.0f);  data[v].tex = vec2f(1.0f, 1.0f);++v;
    // bottom face
    data[v].pos  = vec3f(o.x, o.y, o.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec2f(0.0f, 0.0f); ++v;
    data[v].pos  = vec3f(l.x, o.y, o.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec2f(1.0f, 0.0f); ++v;
    data[v].pos  = vec3f(l.x, o.y, l.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec2f(1.0f, 1.0f); ++v;
    data[v].pos  = vec3f(o.x, o.y, l.z); data[v].nrm = vec3f(0.0f, -1.0f, 0.0f); data[v].tex = vec2f(0.0f, 1.0f); ++v;
}

} // namespace gl
} // namespace scm
