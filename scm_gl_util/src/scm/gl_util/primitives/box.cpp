
#include "box.h"

#include <cassert>

#include <boost/assign/list_of.hpp>

#include <scm/core/memory.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/buffer_objects/vertex_array.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>

namespace scm {
namespace gl {

box_geometry::box_geometry(const render_device_ptr& in_device,
                           const math::vec3f& in_min_vertex,
                           const math::vec3f& in_max_vertex)
  : geometry(in_device),
    _min_vertex(in_min_vertex),
    _max_vertex(in_max_vertex)
{
    using namespace scm::math;

    int num_vertices            = 6 * 4; // 4 vertices per face
    int num_triangles           = 6 * 2; // 2 triangles per face
    int num_triangle_indices    = 3 * num_triangles;
    int num_line_indices        = 2 * 3 * 4; // 12 line segments

    //scoped_array<vec3f>             pos(new vec3f[num_vertices]);
    scoped_array<vec2f>             tex(new vec2f[num_vertices]);
    scoped_array<vec3f>             nrm(new vec3f[num_vertices]);
    scoped_array<vec3f>             tan(new vec3f[num_vertices]);
    scoped_array<vec3f>             btn(new vec3f[num_vertices]);
    scoped_array<unsigned short>    ind_s(new unsigned short[num_triangle_indices]);
    scoped_array<unsigned short>    ind_w(new unsigned short[num_line_indices]);

    const vec3f& o = in_min_vertex;
    const vec3f& l = in_max_vertex;

    _positions = in_device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STREAM_DRAW, num_vertices * sizeof(vec3f), 0);

    render_context_ptr ctx = in_device->main_context();
    vec3f* pos = static_cast<vec3f*>(ctx->map_buffer(_positions, ACCESS_WRITE_INVALIDATE_BUFFER));

    // front face
    int v = 0;
    pos[v]  = vec3f(o.x, o.y, l.z); tex[v] = vec2f(0.0f, 0.0f); nrm[v] = vec3f(0.0f, 0.0f, 1.0f); ++v;
    pos[v]  = vec3f(l.x, o.y, l.z); tex[v] = vec2f(1.0f, 0.0f); nrm[v] = vec3f(0.0f, 0.0f, 1.0f); ++v;
    pos[v]  = vec3f(l.x, l.y, l.z); tex[v] = vec2f(1.0f, 1.0f); nrm[v] = vec3f(0.0f, 0.0f, 1.0f); ++v;
    pos[v]  = vec3f(o.x, l.y, l.z); tex[v] = vec2f(0.0f, 1.0f); nrm[v] = vec3f(0.0f, 0.0f, 1.0f); ++v;
    // right face
    pos[v]  = vec3f(l.x, o.y, l.z); tex[v] = vec2f(0.0f, 0.0f); nrm[v] = vec3f(1.0f, 0.0f, 0.0f); ++v;
    pos[v]  = vec3f(l.x, o.y, o.z); tex[v] = vec2f(1.0f, 0.0f); nrm[v] = vec3f(1.0f, 0.0f, 0.0f); ++v;
    pos[v]  = vec3f(l.x, l.y, o.z); tex[v] = vec2f(1.0f, 1.0f); nrm[v] = vec3f(1.0f, 0.0f, 0.0f); ++v;
    pos[v]  = vec3f(l.x, l.y, l.z); tex[v] = vec2f(0.0f, 1.0f); nrm[v] = vec3f(1.0f, 0.0f, 0.0f); ++v;
    // back face
    pos[v]  = vec3f(l.x, o.y, o.z); tex[v] = vec2f(0.0f, 0.0f); nrm[v] = vec3f(0.0f, 0.0f, -1.0f); ++v;
    pos[v]  = vec3f(o.x, o.y, o.z); tex[v] = vec2f(1.0f, 0.0f); nrm[v] = vec3f(0.0f, 0.0f, -1.0f); ++v;
    pos[v]  = vec3f(o.x, l.y, o.z); tex[v] = vec2f(1.0f, 1.0f); nrm[v] = vec3f(0.0f, 0.0f, -1.0f); ++v;
    pos[v]  = vec3f(l.x, l.y, o.z); tex[v] = vec2f(0.0f, 1.0f); nrm[v] = vec3f(0.0f, 0.0f, -1.0f); ++v;
    // left face
    pos[v]  = vec3f(o.x, o.y, o.z); tex[v] = vec2f(0.0f, 0.0f); nrm[v] = vec3f(-1.0f, 0.0f, 0.0f); ++v;
    pos[v]  = vec3f(o.x, o.y, l.z); tex[v] = vec2f(1.0f, 0.0f); nrm[v] = vec3f(-1.0f, 0.0f, 0.0f); ++v;
    pos[v]  = vec3f(o.x, l.y, l.z); tex[v] = vec2f(1.0f, 1.0f); nrm[v] = vec3f(-1.0f, 0.0f, 0.0f); ++v;
    pos[v]  = vec3f(o.x, l.y, o.z); tex[v] = vec2f(0.0f, 1.0f); nrm[v] = vec3f(-1.0f, 0.0f, 0.0f); ++v;
    // top face
    pos[v]  = vec3f(o.x, l.y, o.z); tex[v] = vec2f(0.0f, 1.0f); nrm[v] = vec3f(0.0f, 1.0f, 0.0f); ++v;
    pos[v]  = vec3f(o.x, l.y, l.z); tex[v] = vec2f(0.0f, 0.0f); nrm[v] = vec3f(0.0f, 1.0f, 0.0f); ++v;
    pos[v]  = vec3f(l.x, l.y, l.z); tex[v] = vec2f(1.0f, 0.0f); nrm[v] = vec3f(0.0f, 1.0f, 0.0f); ++v;
    pos[v]  = vec3f(l.x, l.y, o.z); tex[v] = vec2f(1.0f, 1.0f); nrm[v] = vec3f(0.0f, 1.0f, 0.0f); ++v;
    // bottom face
    pos[v]  = vec3f(o.x, o.y, o.z); tex[v] = vec2f(0.0f, 0.0f); nrm[v] = vec3f(0.0f, -1.0f, 0.0f); ++v;
    pos[v]  = vec3f(l.x, o.y, o.z); tex[v] = vec2f(1.0f, 0.0f); nrm[v] = vec3f(0.0f, -1.0f, 0.0f); ++v;
    pos[v]  = vec3f(l.x, o.y, l.z); tex[v] = vec2f(1.0f, 1.0f); nrm[v] = vec3f(0.0f, -1.0f, 0.0f); ++v;
    pos[v]  = vec3f(o.x, o.y, l.z); tex[v] = vec2f(0.0f, 1.0f); nrm[v] = vec3f(0.0f, -1.0f, 0.0f); ++v;

    ctx->unmap_buffer(_positions);

    assert(v == (6 * 4));

    // solid indices
    for (int f = 0; f < 6; ++f) {
        ind_s[f * 6 + 0] = f * 4 + 0; ind_s[f * 6 + 1] = f * 4 + 1; ind_s[f * 6 + 2] = f * 4 + 2;
        ind_s[f * 6 + 3] = f * 4 + 0; ind_s[f * 6 + 4] = f * 4 + 2; ind_s[f * 6 + 5] = f * 4 + 3;
    }

    // wire indices
    ind_w[0]  =  0; ind_w[1]  =  3;
    ind_w[2]  =  4; ind_w[3]  =  7;
    ind_w[4]  =  8; ind_w[5]  = 11;
    ind_w[6]  = 12; ind_w[7]  = 15;

    ind_w[8]  =  0; ind_w[9]  =  1;
    ind_w[10] =  2; ind_w[11] =  3;
    ind_w[12] =  8; ind_w[13] =  9;
    ind_w[14] = 10; ind_w[15] = 11;

    ind_w[16] =  4; ind_w[17]  = 5;
    ind_w[18] =  6; ind_w[19] =  7;
    ind_w[20] = 12; ind_w[21] = 13;
    ind_w[22] = 14; ind_w[23] = 15;

    using namespace scm::gl;
    using boost::assign::list_of;

//    _positions = in_device->create_buffer(buffer::VERTEX_BUFFER, buffer::STATIC_DRAW, num_vertices * sizeof(vec3f), pos.get());
    _texcoords = in_device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, num_vertices * sizeof(vec2f), tex.get());
    _normals   = in_device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, num_vertices * sizeof(vec3f), nrm.get());

    _solid_indices = in_device->create_buffer(BIND_INDEX_BUFFER, USAGE_STATIC_DRAW, num_triangle_indices * sizeof(unsigned short), ind_s.get());
    _wire_indices  = in_device->create_buffer(BIND_INDEX_BUFFER, USAGE_STATIC_DRAW, num_line_indices * sizeof(unsigned short), ind_w.get());

    _vertex_array = in_device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F)
                                                                (1, 1, TYPE_VEC3F)
                                                                (2, 2, TYPE_VEC2F),
                                                   list_of(_positions)
                                                          (_normals)
                                                          (_texcoords));
}

box_geometry::~box_geometry()
{
    _vertex_array.reset();

    _solid_indices.reset();
    _wire_indices.reset();

    _positions.reset();
    _texcoords.reset();
    _normals.reset();
}

void
box_geometry::update(const render_context_ptr& in_context,
                     const math::vec3f& in_min_vertex,
                     const math::vec3f& in_max_vertex)
{
}

void
box_geometry::draw(const render_context_ptr& in_context,
                   const draw_mode in_draw_mode) const
{
    using namespace scm::gl;
    context_vertex_input_guard vig(in_context);

    in_context->bind_vertex_array(_vertex_array);

    if (in_draw_mode == MODE_SOLID) {
        in_context->bind_index_buffer(_solid_indices, PRIMITIVE_TRIANGLE_LIST, TYPE_USHORT);
        in_context->apply();

        in_context->draw_elements(36);
    }
    else if (in_draw_mode == MODE_WIRE_FRAME) {
        in_context->bind_index_buffer(_wire_indices, PRIMITIVE_LINE_LIST, TYPE_USHORT);
        in_context->apply();

        in_context->draw_elements(24);
    }
}

} // namespace gl
} // namespace scm
