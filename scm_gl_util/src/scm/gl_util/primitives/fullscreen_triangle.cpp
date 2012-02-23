
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "fullscreen_triangle.h"

#include <cassert>

#include <boost/assign/list_of.hpp>

#include <scm/core/memory.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/render_device/context.h>
#include <scm/gl_core/buffer_objects/vertex_array.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>

namespace {
    struct vertex {
        scm::math::vec2f pos;
        scm::math::vec2f tex;
    };
} // namespace

namespace scm {
namespace gl {

fullscreen_triangle::fullscreen_triangle(const render_device_ptr& in_device)
  : geometry(in_device)
{
    using namespace scm::math;

    int num_vertices = 3;

    _vertices = in_device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, num_vertices * sizeof(vertex), 0);

    render_context_ptr ctx = in_device->main_context();
    vertex* data = static_cast<vertex*>(ctx->map_buffer(_vertices, ACCESS_WRITE_INVALIDATE_BUFFER));

#if 0
    data[0].pos = vec3f(0.0f, 0.0f, 0.0f); data[0].tex = vec2f(0.0f, 0.0f);
    data[1].pos = vec3f(2.0f, 0.0f, 0.0f); data[1].tex = vec2f(2.0f, 0.0f);
    data[2].pos = vec3f(0.0f, 2.0f, 0.0f); data[2].tex = vec2f(0.0f, 2.0f);
    
    ctx->unmap_buffer(_vertices);

    using namespace scm::gl;
    using boost::assign::list_of;

    _vertex_array = in_device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F, sizeof(vertex))  // position
                                                                (0, 2, TYPE_VEC2F, sizeof(vertex)), // texture coord
                                                   list_of(_vertices));
#else
    data[0].pos = vec2f(0.0f, 0.0f); data[0].tex = vec2f(0.0f, 0.0f);
    data[1].pos = vec2f(2.0f, 0.0f); data[1].tex = vec2f(2.0f, 0.0f);
    data[2].pos = vec2f(0.0f, 2.0f); data[2].tex = vec2f(0.0f, 2.0f);
    
    ctx->unmap_buffer(_vertices);

    using namespace scm::gl;
    using boost::assign::list_of;

    _vertex_array = in_device->create_vertex_array(vertex_format(0, 0, TYPE_VEC2F, sizeof(vertex))  // position
                                                                (0, 2, TYPE_VEC2F, sizeof(vertex)), // texture coord
                                                   list_of(_vertices));
#endif
}

fullscreen_triangle::~fullscreen_triangle()
{
    _vertex_array.reset();
    _vertices.reset();
}

void
fullscreen_triangle::draw(const render_context_ptr& in_context,
                          const draw_mode in_draw_mode) const
{
    context_vertex_input_guard vig(in_context);

    in_context->bind_vertex_array(_vertex_array);
    in_context->apply();

    if (in_draw_mode == MODE_SOLID) {
        in_context->draw_arrays(PRIMITIVE_TRIANGLE_LIST, 0, 3);
    }
    else if (in_draw_mode == MODE_WIRE_FRAME) {
        in_context->draw_arrays(PRIMITIVE_LINE_LOOP, 0, 3);
    }
    else {
        return;
    }
}

} // namespace gl
} // namespace scm
