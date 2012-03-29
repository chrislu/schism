
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "geometry_highlight.h"

#include <exception>
#include <stdexcept>
#include <string>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>

#include <scm/gl_util/primitives/box.h>

namespace {
std::string wire_v_source = "\
    #version 330 core\n\
    \n\
    uniform mat4  in_mvp;\n\
    \n\
    layout(location = 0) in vec3 in_position;\n\
    \n\
    void main()\n\
    {\n\
        gl_Position  = in_mvp * vec4(in_position, 1.0);\n\
    }\n\
    ";

std::string wire_f_source = "\
    #version 330 core\n\
    \n\
    uniform vec4 in_color;\n\
    layout(location = 0, index = 0) out vec4 out_color;\n\
    void main()\n\
    {\n\
        out_color = in_color;\n\
    }\n\
    ";

std::string solid_v_source = "\
    #version 330 core\n\
    \n\
    uniform mat4  in_mvp;\n\
    uniform mat4  in_mv_it;\n\
    \n\
    out vec3 normal;\n\
    \n\
    layout(location = 0) in vec3 in_position;\n\
    layout(location = 1) in vec3 in_normal;\n\
    \n\
    void main()\n\
    {\n\
        normal       = normalize(in_mv_it * vec4(in_normal, 0.0)).xyz;\n\
        gl_Position  = in_mvp * vec4(in_position, 1.0);\n\
    }\n\
    ";

std::string solid_f_source = "\
    #version 330 core \n\
    \n\
    uniform vec4 in_color;\n\
    \n\
    in vec3 normal;\n\
    \n\
    layout(location = 0, index = 0) out vec4 out_color;\n\
    \n\
    void main()\n\
    {\n\
        //vec3 n = normalize(normal);\n\
        //vec3 l = normalize(vec3(1.0));\n\
        \n\
        //out_color.rgb = in_color.rgb * (dot(n, l) * 0.5 + 0.5);\n\
        //out_color.a = in_color.a;\n\
        out_color = in_color;\n\
    }\n\
    ";

} // namespace

namespace scm {
namespace gl {

geometry_highlight::geometry_highlight(const gl::render_device_ptr& device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    _wire_program = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER, wire_v_source))
                                                  (device->create_shader(STAGE_FRAGMENT_SHADER, wire_f_source)));
    _solid_program = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER, solid_v_source))
                                                  (device->create_shader(STAGE_FRAGMENT_SHADER, solid_f_source)));

    if (   !_wire_program
        || !_solid_program) {
        scm::err() << "geometry_highlight::geometry_highlight(): error creating shader programs." << log::end;
        throw (std::runtime_error("geometry_highlight::geometry_highlight(): error creating shader programs."));
    }

    _blend          = device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO);
    _no_blend       = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _dstate_less    = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _dstate_noz     = device->create_depth_stencil_state(false, false, COMPARISON_LESS);
    _raster_no_cull = device->create_rasterizer_state(FILL_SOLID, CULL_NONE, ORIENT_CCW, true);

    if (   !_no_blend
        || !_dstate_less
        || !_raster_no_cull) {
        scm::err() << "geometry_highlight::geometry_highlight(): error creating state objects." << log::end;
        throw (std::runtime_error("geometry_highlight::geometry_highlight(): error creating state objects."));
    }
}

geometry_highlight::~geometry_highlight()
{
    _wire_program.reset();
    _solid_program.reset();
    _dstate_less.reset();
    _dstate_noz.reset();
    _raster_no_cull.reset();
    _no_blend.reset();
    _blend.reset();
}

void
geometry_highlight::draw(const gl::render_context_ptr& context,
                         const gl::geometry_ptr&       geom,
                         const math::mat4f&            proj_matrix,
                         const math::mat4f&            view_matrix,
                         const gl::geometry::draw_mode dm,
                         const math::vec4f&            color,
                         const float                   line_width)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_state_objects_guard csg(context);
    context_program_guard       cpg(context);
    
    context->set_depth_stencil_state(_dstate_less);
    context->set_rasterizer_state(_raster_no_cull, line_width);

    if (0.0f < color.a) {
        context->set_blend_state(_blend);
    }
    else {
        context->set_blend_state(_no_blend);
    }

    if (dm == gl::geometry::MODE_WIRE_FRAME) {
        _wire_program->uniform("in_mvp", proj_matrix * view_matrix);
        _wire_program->uniform("in_color", color);
        
        context->bind_program(_wire_program);
        geom->draw(context, geometry::MODE_WIRE_FRAME);
    }
    else if (dm == gl::geometry::MODE_SOLID) {
        _solid_program->uniform("in_mvp", proj_matrix * view_matrix);
        _solid_program->uniform("in_mv_it", transpose(inverse(view_matrix)));
        _solid_program->uniform("in_color", color);

        context->bind_program(_solid_program);
        geom->draw(context, geometry::MODE_SOLID);
    }
}

void
geometry_highlight::draw_overlay(const gl::render_context_ptr& context,
                                 const gl::geometry_ptr&       geom,
                                 const math::mat4f&            proj_matrix,
                                 const math::mat4f&            view_matrix,
                                 const gl::geometry::draw_mode dm,
                                 const math::vec4f&            color,
                                 const float                   line_width)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_state_objects_guard csg(context);
    context_program_guard       cpg(context);
    
    context->set_depth_stencil_state(_dstate_noz);
    context->set_rasterizer_state(_raster_no_cull, line_width);

    if (0.0f < color.a) {
        context->set_blend_state(_blend);
    }
    else {
        context->set_blend_state(_no_blend);
    }

    if (dm == gl::geometry::MODE_WIRE_FRAME) {
        _wire_program->uniform("in_mvp", proj_matrix * view_matrix);
        _wire_program->uniform("in_color", color);
        
        context->bind_program(_wire_program);
        geom->draw(context, geometry::MODE_WIRE_FRAME);
    }
    else if (dm == gl::geometry::MODE_SOLID) {
        _solid_program->uniform("in_mvp", proj_matrix * view_matrix);
        _solid_program->uniform("in_mv_it", transpose(inverse(view_matrix)));
        _solid_program->uniform("in_color", color);

        context->bind_program(_solid_program);
        geom->draw(context, geometry::MODE_SOLID);
    }
}

} // namespace gl
} // namespace scm
