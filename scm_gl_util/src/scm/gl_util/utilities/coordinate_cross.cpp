
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "coordinate_cross.h"

#include <exception>
#include <stdexcept>
#include <string>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>

#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>

namespace {
std::string wire_v_source = "\
    #version 330 core\n\
    \n\
    uniform mat4  in_mvp;\n\
    \n\
    out vec3 color;\n\
    \n\
    layout(location = 0) in vec3 in_position;\n\
    layout(location = 3) in vec3 in_color;\n\
    \n\
    void main()\n\
    {\n\
        gl_Position  = in_mvp * vec4(in_position, 1.0);\n\
        color        = in_color;\n\
    }\n\
    ";

std::string wire_f_source = "\
    #version 330 core\n\
    \n\
    uniform vec4 in_color;\n\
    \n\
    in vec3 color;\n\
    \n\
    layout(location = 0) out vec4 out_color;\n\
    void main()\n\
    {\n\
        out_color.rgb = color;\n\
        out_color.a   = 1.0;\n\
    }\n\
    ";

} // namespace

namespace {
    struct vertex {
        scm::math::vec3f pos;
        scm::math::vec3f col;
    };
} // namespace

namespace scm {
namespace gl {

coordinate_cross::coordinate_cross(const gl::render_device_ptr& device,
                                   const float                  line_length)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    size_t  num_vertices = 6 * 2; // 6 lines, 2 vertices each
    
    _vertices = device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STREAM_DRAW, num_vertices * sizeof(vertex), 0);

    render_context_ptr ctx = device->main_context();
    {
        vertex* data = static_cast<vertex*>(ctx->map_buffer(_vertices, ACCESS_WRITE_INVALIDATE_BUFFER));

        if (data) {
            float l = line_length;
            int v = 0;
            // pos x
            data[v].pos = vec3f( 0.0f, 0.0f, 0.0f); data[v].col = vec3f(1.0f, 0.0f, 0.0f); ++v;
            data[v].pos = vec3f( l,    0.0f, 0.0f); data[v].col = vec3f(1.0f, 0.0f, 0.0f); ++v;
            // neg x
            data[v].pos = vec3f( 0.0f, 0.0f, 0.0f); data[v].col = vec3f(0.3f, 0.0f, 0.0f); ++v;
            data[v].pos = vec3f(-l,    0.0f, 0.0f); data[v].col = vec3f(0.3f, 0.0f, 0.0f); ++v;
            // pos y
            data[v].pos = vec3f(0.0f, 0.0f, 0.0f);  data[v].col = vec3f(0.0f, 1.0f, 0.0f); ++v;
            data[v].pos = vec3f(0.0f, l,    0.0f);  data[v].col = vec3f(0.0f, 1.0f, 0.0f); ++v;
            // neg y
            data[v].pos = vec3f(0.0f,  0.0f, 0.0f); data[v].col = vec3f(0.0f, 0.3f, 0.0f); ++v;
            data[v].pos = vec3f(0.0f, -l,    0.0f); data[v].col = vec3f(0.0f, 0.3f, 0.0f); ++v;
            // pos z
            data[v].pos = vec3f(0.0f, 0.0f, 0.0f);  data[v].col = vec3f(0.0f, 0.0f, 1.0f); ++v;
            data[v].pos = vec3f(0.0f, 0.0f, l);     data[v].col = vec3f(0.0f, 0.0f, 1.0f); ++v;
            // neg z
            data[v].pos = vec3f(0.0f, 0.0f,  0.0f); data[v].col = vec3f(0.0f, 0.0f, 0.3f); ++v;
            data[v].pos = vec3f(0.0f, 0.0f, -l);    data[v].col = vec3f(0.0f, 0.0f, 0.3f); ++v;
        }
        else {
            scm::err() << "coordinate_cross::coordinate_cross(): error mapping vertex buffer for vertex update." << log::end;
            throw (std::runtime_error("coordinate_cross::coordinate_cross(): error mapping vertex buffer for vertex update."));
        }

        ctx->unmap_buffer(_vertices);

        _vertex_count  = 12;
        _prim_topology = PRIMITIVE_LINE_LIST;
    }

    _coord_program = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER, wire_v_source))
                                                   (device->create_shader(STAGE_FRAGMENT_SHADER, wire_f_source)));

    _vertex_array = device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F, sizeof(vertex))  // vertex positions
                                                             (0, 3, TYPE_VEC3F, sizeof(vertex)), // vertex colors
                                                list_of(_vertices));

    if (   !_coord_program) {
        scm::err() << "coordinate_cross::coordinate_cross(): error creating shader programs." << log::end;
        throw (std::runtime_error("coordinate_cross::coordinate_cross(): error creating shader programs."));
    }

    _no_blend       = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _dstate_less    = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _dstate_overlay = device->create_depth_stencil_state(false, false, COMPARISON_LESS);
    _raster_no_cull = device->create_rasterizer_state(FILL_SOLID, CULL_NONE, ORIENT_CCW, true);

    if (   !_no_blend
        || !_dstate_less
        || !_raster_no_cull) {
        scm::err() << "coordinate_cross::coordinate_cross(): error creating state objects." << log::end;
        throw (std::runtime_error("coordinate_cross::coordinate_cross(): error creating state objects."));
    }
}

coordinate_cross::~coordinate_cross()
{
    _vertices.reset();
    _vertex_array.reset();
    _coord_program.reset();
    _dstate_less.reset();
    _raster_no_cull.reset();
    _no_blend.reset();
}

void
coordinate_cross::draw(
    const gl::render_context_ptr& context,
    const math::mat4f&            proj_matrix,
    const math::mat4f&            view_matrix,
    const float                   line_width)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_state_objects_guard csg(context);
    context_program_guard       cpg(context);
    context_vertex_input_guard  vig(context);
    
    _coord_program->uniform("in_mvp", proj_matrix * view_matrix);
        
    context->set_depth_stencil_state(_dstate_less);
    context->set_blend_state(_no_blend);
    context->set_rasterizer_state(_raster_no_cull, line_width);

    context->bind_program(_coord_program);
    context->bind_vertex_array(_vertex_array);
    
    context->apply();
    context->draw_arrays(_prim_topology, 0, _vertex_count);
}

void
coordinate_cross::draw_overlay(
    const gl::render_context_ptr& context,
    const math::mat4f&            proj_matrix,
    const math::mat4f&            view_matrix,
    const float                   line_width)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_state_objects_guard csg(context);
    context_program_guard       cpg(context);
    context_vertex_input_guard  vig(context);
    
    _coord_program->uniform("in_mvp", proj_matrix * view_matrix);
        
    context->set_depth_stencil_state(_dstate_overlay);
    context->set_blend_state(_no_blend);
    context->set_rasterizer_state(_raster_no_cull, line_width);

    context->bind_program(_coord_program);
    context->bind_vertex_array(_vertex_array);
    
    context->apply();
    context->draw_arrays(_prim_topology, 0, _vertex_count);
}

} // namespace gl
} // namespace scm
