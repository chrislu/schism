
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

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

#define GEOM_SHADER_FONT  1
#define TWO_PASS_OUTLINES 0

namespace {

#if GEOM_SHADER_FONT == 1
std::string v_source = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    uniform mat4  in_mvp;                                                                           \n\
                                                                                                    \n\
    layout(location = 0) in vec4 in_position_bbox;                                                  \n\
    layout(location = 2) in vec4 in_texcoord_bbox;                                                  \n\
                                                                                                    \n\
    out per_vertex {                                                                                \n\
        vec4 in_position_bbox;                                                                      \n\
        vec4 in_texcoord_bbox;                                                                      \n\
    } v_out;                                                                                        \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        v_out.in_position_bbox  = in_position_bbox;                                                 \n\
        v_out.in_texcoord_bbox  = in_texcoord_bbox;                                                 \n\
        //gl_Position             = in_mvp * vec4(in_position.xy, 0.0, 1.0);                        \n\
    }                                                                                               \n\
    ";

std::string g_source = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    layout(points, invocations = 1)          in;                                                    \n\
    layout(triangle_strip, max_vertices = 4) out;                                                   \n\
                                                                                                    \n\
    uniform mat4  in_mvp;                                                                           \n\
                                                                                                    \n\
    in per_vertex {                                                                                 \n\
        vec4 in_position_bbox;                                                                      \n\
        vec4 in_texcoord_bbox;                                                                      \n\
    } v_in[];                                                                                       \n\
                                                                                                    \n\
    out per_vertex {                                                                                \n\
        vec2 tex_coord;                                                                             \n\
    } v_out;                                                                                        \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        vec2 p  = v_in[0].in_position_bbox.xy;                                                      \n\
        vec2 ps = v_in[0].in_position_bbox.zw;                                                      \n\
                                                                                                    \n\
        vec2 t  = v_in[0].in_texcoord_bbox.xy;                                                      \n\
        vec2 ts = v_in[0].in_texcoord_bbox.zw;                                                      \n\
                                                                                                    \n\
        // 10                                                                                       \n\
        gl_Position       = in_mvp * vec4(p + vec2(ps.x, 0.0), 0.0, 1.0);                           \n\
        v_out.tex_coord   =          vec2(t + vec2(ts.x, 0.0));                                     \n\
        EmitVertex();                                                                               \n\
                                                                                                    \n\
        // 11                                                                                       \n\
        gl_Position       = in_mvp * vec4(p + ps, 0.0, 1.0);                                        \n\
        v_out.tex_coord   =          vec2(t + ts);                                                  \n\
        EmitVertex();                                                                               \n\
                                                                                                    \n\
        // 00                                                                                       \n\
        gl_Position       = in_mvp * vec4(p, 0.0, 1.0);                                             \n\
        v_out.tex_coord   = vec2(t);                                                                \n\
        EmitVertex();                                                                               \n\
                                                                                                    \n\
        // 01                                                                                       \n\
        gl_Position       = in_mvp * vec4(p + vec2(0.0, ps.y), 0.0, 1.0);                           \n\
        v_out.tex_coord   =          vec2(t + vec2(0.0, ts.y));                                     \n\
        EmitVertex();                                                                               \n\
        EndPrimitive();                                                                             \n\
    }                                                                                               \n\
    ";
#else
std::string v_source = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    uniform mat4  in_mvp;                                                                           \n\
                                                                                                    \n\
    layout(location = 0) in vec2 in_position;                                                       \n\
    layout(location = 2) in vec2 in_texcoord;                                                       \n\
                                                                                                    \n\
    out per_vertex {                                                                                \n\
        vec2 tex_coord;                                                                             \n\
    } v_out;                                                                                        \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        //v_out.os_position   = in_position;                                                        \n\
        v_out.tex_coord     = in_texcoord.xy;                                                       \n\
        gl_Position         = in_mvp * vec4(in_position.xy, 0.0, 1.0);                              \n\
    }                                                                                               \n\
    ";

std::string g_source = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    layout(triangles, invocations = 1)       in;                                                    \n\
    layout(triangle_strip, max_vertices = 3) out;                                                   \n\
                                                                                                    \n\
    uniform mat4  in_mvp;                                                                           \n\
                                                                                                    \n\
    in per_vertex {                                                                                 \n\
        vec2 tex_coord;                                                                             \n\
    } v_in[];                                                                                       \n\
                                                                                                    \n\
    out per_vertex {                                                                                \n\
        vec2 tex_coord;                                                                             \n\
    } v_out;                                                                                        \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        for (int i = 0; i < gl_in.length(); ++i) {                                                  \n\
            gl_Position       = gl_in[i].gl_Position;                                               \n\
            //gl_Position       = in_mvp * vec4(v_in[i].os_position, 0.0, 1.0);                     \n\
                                                                                                    \n\
            //v_out.os_position = v_in[i].os_position;                                              \n\
            v_out.tex_coord   = v_in[i].tex_coord;                                                  \n\
            EmitVertex();                                                                           \n\
        }                                                                                           \n\
        EndPrimitive();                                                                             \n\
    }                                                                                               \n\
    ";
#endif
std::string f_source_gray = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    uniform int             in_style;                                                               \n\
    uniform vec4            in_color;                                                               \n\
    uniform sampler2DArray  in_font_array;                                                          \n\
                                                                                                    \n\
    layout(location = 0) out vec4 out_color;                                                        \n\
                                                                                                    \n\
    in per_vertex {                                                                                 \n\
        vec2 tex_coord;                                                                             \n\
    } v_in;                                                                                         \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        float core    = texture(in_font_array,                                                      \n\
                                vec3(v_in.tex_coord, float(in_style))).r;                           \n\
        out_color.rgb = in_color.rgb;                                                               \n\
        out_color.a   = core * in_color.a;                                                          \n\
    }                                                                                               \n\
    ";

std::string f_source_outline_gray = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    uniform int             in_style;                                                               \n\
    uniform vec4            in_color;                                                               \n\
    uniform vec4            in_outline_color;                                                       \n\
    uniform sampler2DArray  in_font_array;                                                          \n\
    uniform sampler2DArray  in_font_border_array;                                                   \n\
                                                                                                    \n\
    layout(location = 0) out vec4 out_color;                                                        \n\
                                                                                                    \n\
    in per_vertex {                                                                                 \n\
        vec2 tex_coord;                                                                             \n\
    } v_in;                                                                                         \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        vec3  tc      = vec3(v_in.tex_coord, float(in_style));                                      \n\
        float core    = texture(in_font_array, tc).r;                                               \n\
        float outline = texture(in_font_border_array, tc).r;                                        \n\
                                                                                                    \n\
        out_color.a   = core + outline - core * outline;                                            \n\
        out_color.rgb =   mix(in_outline_color.rgb * outline, in_color.rgb, core)                   \n\
                        / out_color.a;                                                              \n\
                                                                                                    \n\
        //out_color.rgb = in_color.rgb;                                                             \n\
        //out_color.a   = core * in_color.a;                                                        \n\
    }                                                                                               \n\
    ";

std::string f_source_lcd = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    in vec2 tex_coord;                                                                              \n\
                                                                                                    \n\
    uniform int             in_style;                                                               \n\
    uniform vec4            in_color;                                                               \n\
    uniform sampler2DArray  in_font_array;                                                          \n\
                                                                                                    \n\
    layout(location = 0, index = 0) out vec4 out_color;                                             \n\
    layout(location = 0, index = 1) out vec4 out_sup_pixel_blend;                                   \n\
                                                                                                    \n\
    in per_vertex {                                                                                 \n\
        vec2 tex_coord;                                                                             \n\
    } v_in;                                                                                         \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        vec3 core           = texture(in_font_array,                                                \n\
                                      vec3(v_in.tex_coord, float(in_style))).rgb;                   \n\
                                                                                                    \n\
        out_color           = in_color;                                                             \n\
        out_sup_pixel_blend = vec4(core.rgb * in_color.a, 1.0);                                     \n\
    }                                                                                               \n\
    ";

std::string f_source_outline_lcd = "\
    #version 330 core                                                                               \n\
                                                                                                    \n\
    in vec2 tex_coord;                                                                              \n\
                                                                                                    \n\
    uniform int             in_style;                                                               \n\
    uniform vec4            in_color;                                                               \n\
    uniform vec4            in_outline_color;                                                       \n\
    uniform sampler2DArray  in_font_array;                                                          \n\
    uniform sampler2DArray  in_font_border_array;                                                   \n\
                                                                                                    \n\
    layout(location = 0, index = 0) out vec4 out_color;                                             \n\
    layout(location = 0, index = 1) out vec4 out_sup_pixel_blend;                                   \n\
                                                                                                    \n\
    in per_vertex {                                                                                 \n\
        vec2 tex_coord;                                                                             \n\
    } v_in;                                                                                         \n\
                                                                                                    \n\
    void main()                                                                                     \n\
    {                                                                                               \n\
        vec3 tc      = vec3(v_in.tex_coord, float(in_style));                                       \n\
        vec3 core    = texture(in_font_array, tc).rgb;                                              \n\
        vec3 outline = texture(in_font_border_array, tc).rgb;                                       \n\
                                                                                                    \n\
        out_sup_pixel_blend.rgb = core.rgb + outline.rgb - core.rgb * outline.rgb;                  \n\
        out_color.rgb       =   mix(in_outline_color.rgb * outline.rgb, in_color.rgb, core.rgb)     \n\
                              / out_sup_pixel_blend.rgb;                                            \n\
                                                                                                    \n\
        //out_color           = in_color;                                                           \n\
        //out_sup_pixel_blend = vec4(core.rgb * in_color.a, 1.0);                                   \n\
    }                                                                                               \n\
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

    _font_program_gray = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER,   v_source,      "text_renderer::v_source"))
#if GEOM_SHADER_FONT == 1
                                                       (device->create_shader(STAGE_GEOMETRY_SHADER, g_source,      "text_renderer::g_source"))
#endif
                                                       (device->create_shader(STAGE_FRAGMENT_SHADER, f_source_gray, "text_renderer::f_source_gray")),
                                                "text_renderer::font_program_gray");
    _font_program_lcd  = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER,   v_source,      "text_renderer::v_source"))
#if GEOM_SHADER_FONT == 1
                                                       (device->create_shader(STAGE_GEOMETRY_SHADER, g_source,      "text_renderer::g_source"))
#endif
                                                       (device->create_shader(STAGE_FRAGMENT_SHADER, f_source_lcd,  "text_renderer::f_source_lcd")),
                                                "text_renderer::font_program_lcd");
    _font_program_outline_gray = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER,   v_source,      "text_renderer::v_source"))
#if GEOM_SHADER_FONT == 1
                                                               (device->create_shader(STAGE_GEOMETRY_SHADER, g_source,      "text_renderer::g_source"))
#endif
                                                               (device->create_shader(STAGE_FRAGMENT_SHADER, f_source_outline_gray, "text_renderer::f_source_outline_gray")),
                                                "text_renderer::font_program_outline_gray");
    _font_program_outline_lcd  = device->create_program(list_of(device->create_shader(STAGE_VERTEX_SHADER,   v_source,      "text_renderer::v_source"))
#if GEOM_SHADER_FONT == 1
                                                               (device->create_shader(STAGE_GEOMETRY_SHADER, g_source,      "text_renderer::g_source"))
#endif
                                                               (device->create_shader(STAGE_FRAGMENT_SHADER, f_source_outline_lcd,  "text_renderer::f_source_outline_lcd")),
                                                "text_renderer::font_program_outline_lcd");

    if (   !_font_program_gray
        || !_font_program_lcd
        || !_font_program_outline_gray
        || !_font_program_outline_lcd) {
        scm::err() << "font_renderer::font_renderer(): error creating shader programs." << log::end;
        throw std::runtime_error("font_renderer::font_renderer(): error creating shader programs.");
    }

    _font_sampler_state = device->create_sampler_state(FILTER_MIN_MAG_NEAREST, WRAP_CLAMP_TO_EDGE);
    _font_blend_gray    = device->create_blend_state(true, FUNC_SRC_ALPHA,  FUNC_ONE_MINUS_SRC_ALPHA,  FUNC_ONE, FUNC_ZERO);
    _font_blend_lcd     = device->create_blend_state(true, FUNC_SRC1_COLOR, FUNC_ONE_MINUS_SRC1_COLOR, FUNC_ONE, FUNC_ZERO);
    //_font_blend_lcd     = device->create_blend_state(true, FUNC_ONE, FUNC_ZERO, FUNC_ONE, FUNC_ZERO);
    _font_dstate        = device->create_depth_stencil_state(false, false, COMPARISON_LESS);
    _font_raster_state  = device->create_rasterizer_state(FILL_SOLID, CULL_BACK, ORIENT_CCW, true);

    if (   !_font_sampler_state
        || !_font_blend_gray
        || !_font_blend_lcd
        || !_font_dstate
        || !_font_raster_state) {
        scm::err() << "font_renderer::font_renderer(): error creating state objects." << log::end;
        throw std::runtime_error("font_renderer::font_renderer(): error creating state objects.");
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
    
    mat4f v = make_translation(vec3f(vec2f(pos), 0.0f));
    //scale(v, static_cast<float>(txt->font()->styles_texture_array()->dimensions().x),
    //         static_cast<float>(txt->font()->styles_texture_array()->dimensions().y), 1.0f);
    mat4f mvp = _projection_matrix * v;

    context->set_depth_stencil_state(_font_dstate);
    context->set_rasterizer_state(_font_raster_state);
    context->bind_texture(txt->font()->styles_texture_array(), _font_sampler_state, 0);

    switch (txt->font()->smooth_style()) {
        case font_face::smooth_normal:
            _font_program_gray->uniform("in_mvp", mvp);
            _font_program_gray->uniform("in_style", static_cast<int>(txt->text_style()));
            _font_program_gray->uniform("in_color", txt->text_color());
            _font_program_gray->uniform_sampler("in_font_array", 0);

            context->set_blend_state(_font_blend_gray);
            context->bind_program(_font_program_gray);
           break;
        case font_face::smooth_lcd:
            _font_program_lcd->uniform("in_mvp", mvp);
            _font_program_lcd->uniform("in_style", static_cast<int>(txt->text_style()));
            _font_program_lcd->uniform("in_color", txt->text_color());
            _font_program_lcd->uniform_sampler("in_font_array", 0);

            context->set_blend_state(_font_blend_lcd/*, txt->text_color()*/);
            context->bind_program(_font_program_lcd);
            break;
        default:
            return;
    }

#if GEOM_SHADER_FONT == 1
    if (txt->_indices_count > 0) {
        context->bind_vertex_array(txt->_vertex_array);
        context->apply();
        context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
    }
#else
    if (txt->_indices_count > 0) {
        context->bind_vertex_array(txt->_vertex_array);
        context->bind_index_buffer(txt->_index_buffer, txt->_topology, TYPE_USHORT);
        context->apply();
        context->draw_elements(txt->_indices_count);
    }
#endif

    //_quad->draw(context, geometry::MODE_SOLID);
}

void
text_renderer::draw_outlined(const render_context_ptr& context,
                             const math::vec2i&        pos,
                             const text_ptr&           txt) const
{
    if (!txt->font()->styles_border_texture_array()) {
        return draw(context, pos, txt);
    }

    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    context_vertex_input_guard  vig(context);
    context_state_objects_guard csg(context);
    context_texture_units_guard tug(context);
    context_program_guard       cpg(context);

    context->set_depth_stencil_state(_font_dstate);
    context->set_rasterizer_state(_font_raster_state);

    context->bind_vertex_array(txt->_vertex_array);
#if GEOM_SHADER_FONT == 1
#else
    context->bind_index_buffer(txt->_index_buffer, txt->_topology, TYPE_USHORT);
#endif

    mat4f  v  = make_translation(vec3f(vec2f(pos), 0.0f));
    mat4f mvp = _projection_matrix * v;

    switch (txt->font()->smooth_style()) {
        case font_face::smooth_normal:
            _font_program_outline_gray->uniform("in_mvp",               mvp);
            _font_program_outline_gray->uniform("in_style",             static_cast<int>(txt->text_style()));
            _font_program_outline_gray->uniform("in_color",             txt->text_color());
            _font_program_outline_gray->uniform("in_outline_color",     txt->text_outline_color());
            _font_program_outline_gray->uniform_sampler("in_font_array",        0);
            _font_program_outline_gray->uniform_sampler("in_font_border_array", 1);

            context->set_blend_state(_font_blend_gray);
            context->bind_texture(txt->font()->styles_texture_array(),        _font_sampler_state, 0);
            context->bind_texture(txt->font()->styles_border_texture_array(), _font_sampler_state, 1);
            context->bind_program(_font_program_outline_gray);
#if TWO_PASS_OUTLINES != 1
#if GEOM_SHADER_FONT == 1
            if (txt->_indices_count > 0) {
                context->apply();
                context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
            }
#else
            if (txt->_indices_count > 0) {
                context->apply();
                context->draw_elements(txt->_indices_count);
            }
#endif
#else
            if (txt->font()->styles_border_texture_array()) { // outline
                _font_program_outline_gray->uniform("in_color",         txt->text_outline_color());
                context->bind_texture(txt->font()->styles_texture_array(),        _font_sampler_state, 1);
                context->bind_texture(txt->font()->styles_border_texture_array(), _font_sampler_state, 0);

#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
            }
            { // text
                _font_program_outline_gray->uniform("in_color", txt->text_color());
                context->bind_texture(txt->font()->styles_texture_array(),        _font_sampler_state, 0);
                context->bind_texture(txt->font()->styles_border_texture_array(), _font_sampler_state, 1);

#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
            }
#endif
            break;
        case font_face::smooth_lcd:
            _font_program_outline_lcd->uniform("in_mvp",               mvp);
            _font_program_outline_lcd->uniform("in_style",             static_cast<int>(txt->text_style()));
            _font_program_outline_lcd->uniform("in_color",             txt->text_color());
            _font_program_outline_lcd->uniform("in_outline_color",     txt->text_outline_color());
            _font_program_outline_lcd->uniform_sampler("in_font_array",        0);
            _font_program_outline_lcd->uniform_sampler("in_font_border_array", 1);

            context->set_blend_state(_font_blend_lcd);
            context->bind_texture(txt->font()->styles_texture_array(),        _font_sampler_state, 0);
            context->bind_texture(txt->font()->styles_border_texture_array(), _font_sampler_state, 1);
            context->bind_program(_font_program_outline_lcd);

#if TWO_PASS_OUTLINES != 1
#if GEOM_SHADER_FONT == 1
            if (txt->_indices_count > 0) {
                context->apply();
                context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
            }
#else
            if (txt->_indices_count > 0) {
                context->apply();
                context->draw_elements(txt->_indices_count);
            }
#endif
#else
            if (txt->font()->styles_border_texture_array()) { // outline
                context->bind_texture(txt->font()->styles_border_texture_array(), _font_sampler_state, 0);
                context->bind_texture(txt->font()->styles_texture_array(),        _font_sampler_state, 1);

                _font_program_outline_lcd->uniform("in_color", txt->text_outline_color());


#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
            }
            { // text
                context->bind_texture(txt->font()->styles_texture_array(),        _font_sampler_state, 0);
                context->bind_texture(txt->font()->styles_border_texture_array(), _font_sampler_state, 1);

                _font_program_outline_lcd->uniform("in_color", txt->text_color());


#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
            }
#endif
            break;
        default:
            return;
    }

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
#if GEOM_SHADER_FONT == 1
#else
    context->bind_index_buffer(txt->_index_buffer, txt->_topology, TYPE_USHORT);
#endif

    switch (txt->font()->smooth_style()) {
        case font_face::smooth_normal:
            _font_program_gray->uniform_sampler("in_font_array", 0);
            context->set_blend_state(_font_blend_gray);
            context->bind_program(_font_program_gray);
            { // shadow
                mat4f v   = make_translation(vec3f(vec2f(pos + txt->text_shadow_offset()), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_gray->uniform("in_mvp", mvp);
                _font_program_gray->uniform("in_style", static_cast<int>(txt->text_style()));
                _font_program_gray->uniform("in_color", txt->text_shadow_color());

#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
            }
            { // text
                mat4f v = mat4f::identity();
                translate(v, vec3f(vec2f(pos), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_gray->uniform("in_mvp", mvp);
                _font_program_gray->uniform("in_style", static_cast<int>(txt->text_style()));
                _font_program_gray->uniform("in_color", txt->text_color());

#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
            }
            break;
        case font_face::smooth_lcd:
            _font_program_lcd->uniform_sampler("in_font_array", 0);
            context->bind_program(_font_program_lcd);
            { // shadow
                mat4f v   = make_translation(vec3f(vec2f(pos + txt->text_shadow_offset()), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_lcd->uniform("in_mvp", mvp);
                _font_program_lcd->uniform("in_style", static_cast<int>(txt->text_style()));
                _font_program_lcd->uniform("in_color", txt->text_shadow_color());
                context->set_blend_state(_font_blend_lcd/*, txt->text_shadow_color()*/);

#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
            }
            { // text
                mat4f v = mat4f::identity();
                translate(v, vec3f(vec2f(pos), 0.0f));
                mat4f mvp = _projection_matrix * v;

                _font_program_lcd->uniform("in_mvp", mvp);
                _font_program_lcd->uniform("in_style", static_cast<int>(txt->text_style()));
                _font_program_lcd->uniform("in_color", txt->text_color());
                context->set_blend_state(_font_blend_lcd/*, txt->text_color()*/);

#if GEOM_SHADER_FONT == 1
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_arrays(PRIMITIVE_POINT_LIST, 0, txt->_indices_count);
                }
#else
                if (txt->_indices_count > 0) {
                    context->apply();
                    context->draw_elements(txt->_indices_count);
                }
#endif
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
