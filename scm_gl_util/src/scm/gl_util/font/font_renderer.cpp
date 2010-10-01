
#include "font_renderer.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>

#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/texture_objects.h>
#include <scm/gl_core/shader_objects.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/primitives/quad.h>

namespace {
std::string v_source = "\
    #version 330\n\
    \
    out vec2 tex_coord;\
    \
    uniform mat4  in_mvp;\
    \
    layout(location = 0) in vec3 in_position;\
    layout(location = 2) in vec3 in_texcoord;\
    \
    void main()\
    {\
        tex_coord    = in_texcoord.xy;\
        gl_Position  = in_mvp * vec4(in_position, 1.0);\
    }\
    ";

std::string f_source = "\
    #version 330\n\
    \
    in vec2 tex_coord;\
    \
    uniform vec4            in_color;\
    uniform sampler2DArray  in_font_array;\
    uniform int             in_style;\
    \
    layout(location = 0) out vec4 out_color;\
    \
    void main()\
    {\
        float core = texture(in_font_array, vec3(tex_coord, float(in_style))).r;\
        out_color = in_color * vec4(core);\
    }\
    ";

} // namespace


namespace scm {
namespace gl {

font_renderer::font_renderer(const render_device_ptr& device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    _font_shader_program = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER, v_source))
                                                         (device->create_shader(STAGE_FRAGMENT_SHADER, f_source)));

    if (!_font_shader_program) {
        scm::err() << "font_renderer::font_renderer(): error creating shader program." << log::end;
        throw (std::runtime_error("font_renderer::font_renderer(): error creating shader program."));
    }

    _font_sampler_state = device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
    _font_blend_state   = device->create_blend_state(false, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _font_dstate        = device->create_depth_stencil_state(false, false, COMPARISON_LESS);
    _font_raster_state  = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);

    if (   !_font_sampler_state
        || !_font_blend_state
        || !_font_dstate
        || !_font_raster_state) {
        scm::err() << "font_renderer::font_renderer(): error creating state objects." << log::end;
        throw (std::runtime_error("font_renderer::font_renderer(): error creating state objects."));
    }

    _quad.reset(new quad_geometry(device, vec2f(0.0f), vec2f(1.0f), vec2f(0.0f), vec2f(1.0f)));
}

font_renderer::~font_renderer()
{
    _font_shader_program.reset();
    _font_sampler_state.reset();
    _font_dstate.reset();
    _font_raster_state.reset();
    _font_blend_state.reset();

    _quad.reset();
}

void
font_renderer::draw(const render_context_ptr& context,
                    const font_face_cptr&     font) const
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_state_objects_guard csg(context);
    context_texture_units_guard tug(context);
    context_program_guard       cpg(context);
    
    mat4f p = mat4f::identity();
    ortho_matrix(p, -960.0f, 960.0f, -600.0f, 600.0f, -1.0f, 1.0f);
    mat4f v = mat4f::identity();
    scale(v, static_cast<float>(font->styles_texture_array()->dimensions().x),
             static_cast<float>(font->styles_texture_array()->dimensions().y), 1.0f);
    mat4f mvp = p * v;

    _font_shader_program->uniform("in_mvp", mvp);
    _font_shader_program->uniform("in_font_array", 0);
    _font_shader_program->uniform("in_color", vec4f(1.0f));
    _font_shader_program->uniform("in_style", 0);

    context->set_depth_stencil_state(_font_dstate);
    context->set_rasterizer_state(_font_raster_state);
    context->set_blend_state(_font_blend_state);

    context->bind_texture(font->styles_texture_array(), _font_sampler_state, 0);
    context->bind_program(_font_shader_program);

    _quad->draw(context, geometry::MODE_SOLID);
}

} // namespace gl
} // namespace scm
