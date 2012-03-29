
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "draw_helpers.h"

#include <exception>
#include <stdexcept>
#include <string>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/shader_objects.h>
#include <scm/gl_core/state_objects.h>

#include <scm/gl_util/primitives/quad.h>

namespace {
std::string v_source = "\
    #version 330\n\
    \
    uniform mat4  in_mvp;\
    \
    layout(location = 0) in vec3 in_position;\
    \
    void main()\
    {\
        gl_Position  = in_mvp * vec4(in_position, 1.0);\
    }\
    ";

std::string f_source = "\
    #version 330\n\
    \
    uniform vec4 in_color;\
    layout(location = 0) out vec4 out_color;\
    void main()\
    {\
        out_color = in_color;\
    }\
    ";

} // namespace

namespace scm {
namespace data {

quad_highlight::quad_highlight(const gl::render_device_ptr& device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    _program = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER, v_source))
                                             (device->create_shader(STAGE_FRAGMENT_SHADER, f_source)));

    if (!_program) {
        scm::err() << "quad_highlight::quad_highlight(): error creating shader program." << log::end;
        throw (std::runtime_error("quad_highlight::quad_highlight(): error creating shader program."));
    }

    _no_blend       = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _dstate_less    = device->create_depth_stencil_state(true, true, COMPARISON_LESS);
    _raster_no_cull = device->create_rasterizer_state(FILL_SOLID, CULL_NONE, ORIENT_CCW, true);

    if (   !_no_blend
        || !_dstate_less
        || !_raster_no_cull) {
        scm::err() << "quad_highlight::quad_highlight(): error creating state objects." << log::end;
        throw (std::runtime_error("quad_highlight::quad_highlight(): error creating state objects."));
    }
}

quad_highlight::~quad_highlight()
{
    _program.reset();
    _dstate_less.reset();
    _raster_no_cull.reset();
    _no_blend.reset();
}

void
quad_highlight::draw(const gl::render_context_ptr& context,
                     const gl::quad_geometry&      quad,
                     const math::mat4f&            mvp,
                     const math::vec4f&            color,
                     const float                   line_width)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_state_objects_guard csg(context);
    context_program_guard       cpg(context);
    
    _program->uniform("in_mvp", mvp);
    _program->uniform("in_color", color);

    context->set_depth_stencil_state(_dstate_less);
    context->set_blend_state(_no_blend);
    context->set_rasterizer_state(_raster_no_cull, line_width);

    context->bind_program(_program);

    quad.draw(context, geometry::MODE_WIRE_FRAME);
}

} // namespace data
} // namespace scm
