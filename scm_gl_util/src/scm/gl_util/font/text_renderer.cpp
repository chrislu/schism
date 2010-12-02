
#include "text_renderer.h"

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
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/primitives/quad.h>

namespace {
std::string v_source = "\
    #version 330 core\n\
    \n\
    out vec3 tex_coord;\n\
    \n\
    uniform mat4  in_mvp;\n\
    \n\
    layout(location = 0) in vec3 in_position;\n\
    layout(location = 2) in vec3 in_texcoord;\n\
    \n\
    void main()\n\
    {\n\
        tex_coord    = in_texcoord.xyz;\n\
        gl_Position  = in_mvp * vec4(in_position, 1.0);\n\
    }\n\
    ";

std::string f_source_gray = "\
    #version 330 core\n\
    \n\
    in vec3 tex_coord;\n\
    \n\
    uniform vec4            in_color;\n\
    uniform sampler2DArray  in_font_array;\n\
    \n\
    layout(location = 0) out vec4 out_color;\n\
    \n\
    void main()\n\
    {\n\
        float core = texture(in_font_array, tex_coord).r;\n\
        out_color.rgb = in_color.rgb;\n\
        out_color.a   = core * in_color.a;\n\
    }\n\
    ";

std::string f_source_lcd = "\
    #version 330 core\n\
    \n\
    in vec3 tex_coord;\n\
    \n\
    uniform vec4            in_color;\n\
    uniform sampler2DArray  in_font_array;\n\
    \n\
    layout(location = 0) out vec4 out_color;\n\
    \n\
    void main()\n\
    {\n\
        vec3 core = texture(in_font_array, tex_coord).rgb;\n\
        out_color.rgb = core.rgb * in_color.a;\n\
        out_color.a   = 1.0;\n\
    }\n\
    ";

} // namespace


namespace scm {
namespace gl {

text_renderer::text_renderer(const render_device_ptr& device)
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;
    using boost::assign::list_of;

    _font_program_gray = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER, v_source,        "text_renderer::v_source"))
                                                       (device->create_shader(STAGE_FRAGMENT_SHADER, f_source_gray, "text_renderer::f_source_gray")),
                                                "text_renderer::font_program_gray");
    _font_program_lcd  = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER, v_source,        "text_renderer::v_source"))
                                                       (device->create_shader(STAGE_FRAGMENT_SHADER, f_source_lcd,  "text_renderer::f_source_lcd")),
                                                "text_renderer::font_program_lcd");

    if (   !_font_program_gray
        || !_font_program_lcd) {
        scm::err() << "font_renderer::font_renderer(): error creating shader programs." << log::end;
        throw (std::runtime_error("font_renderer::font_renderer(): error creating shader programs."));
    }

    _font_sampler_state = device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
    _font_blend_gray    = device->create_blend_state(true, FUNC_SRC_ALPHA, FUNC_ONE_MINUS_SRC_ALPHA, FUNC_ONE, FUNC_ZERO);
    _font_blend_lcd     = device->create_blend_state(true, FUNC_CONSTANT_COLOR, FUNC_ONE_MINUS_SRC_COLOR, FUNC_ONE, FUNC_ZERO);
    //_font_blend_lcd     = device->create_blend_state(true, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _font_dstate        = device->create_depth_stencil_state(false, false, COMPARISON_LESS);
    _font_raster_state  = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);

    if (   !_font_sampler_state
        || !_font_blend_gray
        || !_font_blend_lcd
        || !_font_dstate
        || !_font_raster_state) {
        scm::err() << "font_renderer::font_renderer(): error creating state objects." << log::end;
        throw (std::runtime_error("font_renderer::font_renderer(): error creating state objects."));
    }

    //_quad.reset(new quad_geometry(device, vec2f(0.0f), vec2f(1.0f), vec2f(0.0f), vec2f(1.0f)));
}

text_renderer::~text_renderer()
{
    _font_program_gray.reset();
    _font_program_lcd.reset();
    _font_sampler_state.reset();
    _font_dstate.reset();
    _font_raster_state.reset();
    _font_blend_gray.reset();
    _font_blend_lcd.reset();

    //_quad.reset();
}

void
text_renderer::draw(const render_context_ptr& context,
                    const math::vec2i&        pos,
                    const text_ptr&           txt) const
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_vertex_input_guard  vig(context);
    context_state_objects_guard csg(context);
    context_texture_units_guard tug(context);
    context_program_guard       cpg(context);
    
    mat4f v = mat4f::identity();
    translate(v, vec3f(vec2f(pos), 0.0f));
    //scale(v, static_cast<float>(txt->font()->styles_texture_array()->dimensions().x),
    //         static_cast<float>(txt->font()->styles_texture_array()->dimensions().y), 1.0f);
    mat4f mvp = _projection_matrix * v;

    context->set_depth_stencil_state(_font_dstate);
    context->set_rasterizer_state(_font_raster_state);
    context->bind_texture(txt->font()->styles_texture_array(), _font_sampler_state, 0);

    switch (txt->font()->smooth_style()) {
        case font_face::smooth_normal:
            _font_program_gray->uniform("in_mvp", mvp);
            _font_program_gray->uniform("in_font_array", 0);
            _font_program_gray->uniform("in_color", txt->text_color());

            context->set_blend_state(_font_blend_gray);
            context->bind_program(_font_program_gray);
           break;
        case font_face::smooth_lcd:
            _font_program_lcd->uniform("in_mvp", mvp);
            _font_program_lcd->uniform("in_font_array", 0);
            _font_program_lcd->uniform("in_color", txt->text_color());

            context->set_blend_state(_font_blend_lcd, txt->text_color());
            context->bind_program(_font_program_lcd);
            break;
        default:
            return;
    }

    context->bind_vertex_array(txt->_vertex_array);
    context->bind_index_buffer(txt->_index_buffer, txt->_topology, TYPE_USHORT);
    context->apply();
    context->draw_elements(txt->_indices_count);

    //_quad->draw(context, geometry::MODE_SOLID);
}

void
text_renderer::draw_shadowed(const render_context_ptr& context,
                             const math::vec2i&        pos,
                             const text_ptr&           txt) const
{
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_vertex_input_guard  vig(context);
    context_state_objects_guard csg(context);
    context_texture_units_guard tug(context);
    context_program_guard       cpg(context);
    

    context->set_depth_stencil_state(_font_dstate);
    context->set_rasterizer_state(_font_raster_state);
    context->bind_texture(txt->font()->styles_texture_array(), _font_sampler_state, 0);

    context->bind_vertex_array(txt->_vertex_array);
    context->bind_index_buffer(txt->_index_buffer, txt->_topology, TYPE_USHORT);

    switch (txt->font()->smooth_style()) {
        case font_face::smooth_normal:
            _font_program_gray->uniform("in_font_array", 0);
            context->set_blend_state(_font_blend_gray);
            context->bind_program(_font_program_gray);
            { // shadow
                mat4f v = mat4f::identity();
                translate(v, vec3f(vec2f(pos + txt->text_shadow_offset()), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_gray->uniform("in_mvp", mvp);
                _font_program_gray->uniform("in_color", txt->text_shadow_color());

                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
            }
            { // text
                mat4f v = mat4f::identity();
                translate(v, vec3f(vec2f(pos), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_gray->uniform("in_mvp", mvp);
                _font_program_gray->uniform("in_color", txt->text_color());

                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
            }
            break;
        case font_face::smooth_lcd:
            _font_program_lcd->uniform("in_font_array", 0);
            context->bind_program(_font_program_lcd);
            { // shadow
                mat4f v = mat4f::identity();
                translate(v, vec3f(vec2f(pos + txt->text_shadow_offset()), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_lcd->uniform("in_mvp", mvp);
                _font_program_lcd->uniform("in_color", txt->text_shadow_color());
                context->set_blend_state(_font_blend_lcd, txt->text_shadow_color());

                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
            }
            { // text
                mat4f v = mat4f::identity();
                translate(v, vec3f(vec2f(pos), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_lcd->uniform("in_mvp", mvp);
                _font_program_lcd->uniform("in_color", txt->text_color());
                context->set_blend_state(_font_blend_lcd, txt->text_color());

                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
            }
            break;
        default:
            return;
    }

    //_quad->draw(context, geometry::MODE_SOLID);
}

void
text_renderer::projection_matrix(const math::mat4f& m)
{
    _projection_matrix = m;
}

} // namespace gl
} // namespace scm
