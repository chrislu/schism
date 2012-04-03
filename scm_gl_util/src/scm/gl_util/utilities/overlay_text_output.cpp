
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "overlay_text_output.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>

#include <scm/gl_core/math.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/state_objects.h>
#include <scm/gl_core/texture_objects.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>
#include <scm/gl_util/primitives/quad.h>
#include <scm/gl_util/utilities/geometry_highlight.h>

namespace scm {
namespace gl {
namespace util {

overlay_text_output::overlay_text_output(const gl::render_device_ptr& device,
                                         const math::vec2ui&          vp_size,
                                         const int                    text_size)
  : _viewport_size(vp_size)
  , _back_color(math::vec4f(0.7f, 0.7f, 0.7f, 0.8f))
  , _frame_color(math::vec4f::one())
  , _output_text_size(text_size)
  , _output_text_pos(math::vec2i(5, vp_size.y - (text_size + 2)))
  , _output_text_frame_size(math::vec2i(5))
{
    using namespace scm::math;

    try {
        //font_face_ptr output_font(new font_face(device, "../../../res/fonts/Segoeui.ttf", 48, 6.0f, font_face::smooth_lcd));
        //font_face_ptr output_font(new font_face(device, "../../../res/fonts/Consola.ttf", 12, 0.5f, font_face::smooth_lcd));
        font_face_ptr output_font(new font_face(device, "../../../res/fonts/UbuntuMono.ttf", _output_text_size, 0.8f, font_face::smooth_lcd));
        _text_renderer  = make_shared<text_renderer>(device);
        _output_text    = make_shared<text>(device, output_font, font_face::style_regular, "sick, sad world...");

        mat4f   fs_projection = make_ortho_matrix(0.0f, static_cast<float>(_viewport_size.x),
                                                  0.0f, static_cast<float>(_viewport_size.y), -1.0f, 1.0f);
        _text_renderer->projection_matrix(fs_projection);

        //_output_text->text_color(math::vec4f(0.8f, 0.66f, 0.41f, 1.0f));
        //_output_text->text_outline_color(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f));
        _output_text->text_color(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f));
        _output_text->text_outline_color(math::vec4f(0.75f, 0.75f, 0.75f, 1.0f));
        _output_text->text_kerning(true);

        _output_text_background = scm::make_shared<gl::quad_geometry>(device, math::vec2f(_output_text_pos), math::vec2f(_output_text_pos) + math::vec2f(_output_text->text_bounding_box()));
        _geom_highlighter       = scm::make_shared<gl::geometry_highlight>(device);
    }
    catch(const std::exception& e) {
        throw std::runtime_error(std::string("overlay_text_output::overlay_text_output(): ") + e.what());
    }
}

overlay_text_output::~overlay_text_output()
{
    _geom_highlighter.reset();
    _output_text_background.reset();
    _output_text.reset();
    _text_renderer.reset();
}

const math::vec4f&
overlay_text_output::back_color() const
{
    return _back_color;
}

void
overlay_text_output::back_color(const math::vec4f& c)
{
    _back_color = c;
}

const math::vec4f&
overlay_text_output::frame_color() const
{
    return _frame_color;
}

void
overlay_text_output::frame_color(const math::vec4f& c)
{
    _frame_color = c;
}

const math::vec4f&
overlay_text_output::text_color() const
{
    return _output_text->text_color();
}

void
overlay_text_output::text_color(const math::vec4f& c)
{
    _output_text->text_color(c);
}

const math::vec4f&
overlay_text_output::text_outline_color() const
{
    return _output_text->text_outline_color();
}

void
overlay_text_output::text_outline_color(const math::vec4f& c)
{
    _output_text->text_outline_color(c);
}

void
overlay_text_output::update(const gl::render_context_ptr& context,
                            const std::string&            text,
                            const math::vec2i&            out_pos)
{
    using namespace scm::math;

    _output_text_pos = out_pos;
    update(context, text);
}

void
overlay_text_output::update(const gl::render_context_ptr& context,
                            const std::string&            text)
{
    using namespace scm::math;

    _output_text->text_string(text);
    vec2i   ll = vec2i(_output_text_pos.x - _output_text_frame_size.x,
                       _output_text_pos.y + _output_text_frame_size.y + _output_text_size - _output_text->text_bounding_box().y - _output_text_frame_size.y);
    vec2i   ur = vec2i(_output_text_pos.x + _output_text->text_bounding_box().x + _output_text_frame_size.x,
                       _output_text_pos.y + _output_text_size + _output_text_frame_size.y);
    _output_text_background->update(context, vec2f(ll), vec2f(ur));
}

void
overlay_text_output::draw(const gl::render_context_ptr& context)
{
    using namespace scm::math;

    mat4f fs_proj = make_ortho_matrix(0.0f, static_cast<float>(_viewport_size.x), 0.0f, static_cast<float>(_viewport_size.y), -1.0f, 1.0f);
    _geom_highlighter->draw_overlay(context, _output_text_background, fs_proj, mat4f::identity(), geometry::MODE_SOLID, _back_color);
    _geom_highlighter->draw_overlay(context, _output_text_background, fs_proj, mat4f::identity(), geometry::MODE_WIRE_FRAME, _frame_color);
    _text_renderer->draw_outlined(context, _output_text_pos, _output_text);
}

} // namespace util
} // namespace gl
} // namespace scm
