
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "text.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <exception>
#include <stdexcept>
#include <string>
#include <sstream>

#include <boost/assign/list_of.hpp>

#include <scm/log.h>

#include <scm/gl_core/math.h>
#include <scm/gl_core/buffer_objects.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects/scoped_buffer_map.h>

#include <scm/gl_util/font/font_face.h>

#define GEOM_SHADER_FONT 1

namespace {
struct vertex {
#if GEOM_SHADER_FONT == 1
    scm::math::vec4f pos_bbox;
    scm::math::vec4f tex_bbox;
#else
    scm::math::vec2f pos;
    scm::math::vec2f tex;
#endif
};
} // namespace

namespace scm {
namespace gl {

text::text(const render_device_ptr&     device,
           const font_face_cptr&        font,
           const font_face::style_type  stl,
           const std::string&           str)
  : _font(font)
  , _text_style(stl)
  , _text_string(str)
  , _text_kerning(true)
  , _text_color(math::vec4f::one())
  , _text_outline_color(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f))
  , _text_shadow_color(math::vec4f(0.0f, 0.0f, 0.0f, 1.0f))
  , _text_shadow_offset(math::vec2i(1, -1))
  , _text_bounding_box(math::vec2i(0, 0))
  , _indices_count(0)
  , _topology(PRIMITIVE_TRIANGLE_LIST)
  , _glyph_capacity(20)
  , _render_device(device)
  , _render_context(device->main_context())
{
    using boost::assign::list_of;

#if GEOM_SHADER_FONT == 1
    int num_vertices = _glyph_capacity; // one point per glyph 
    _vertex_buffer = device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STREAM_DRAW, num_vertices * sizeof(vertex), 0);
    _vertex_array  = device->create_vertex_array(vertex_format(0, 0, TYPE_VEC4F, sizeof(vertex))
                                                              (0, 2, TYPE_VEC4F, sizeof(vertex)),
                                                 list_of(_vertex_buffer));
#else
    int num_vertices = _glyph_capacity * 4; // one quad per glyph 
    int num_indices  = _glyph_capacity * 6; // two triangles per glyph
    _vertex_buffer = device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STREAM_DRAW, num_vertices * sizeof(vertex), 0);
    _index_buffer  = device->create_buffer(BIND_INDEX_BUFFER, USAGE_STREAM_DRAW,  num_indices  * sizeof(unsigned short), 0);
    _vertex_array  = device->create_vertex_array(vertex_format(0, 0, TYPE_VEC2F, sizeof(vertex))
                                                              (0, 2, TYPE_VEC2F, sizeof(vertex)),
                                                 list_of(_vertex_buffer));

    // fill index data
    unsigned short* index_data  = static_cast<unsigned short*>(device->main_context()->map_buffer(_index_buffer, ACCESS_WRITE_INVALIDATE_BUFFER));

    if (0 == index_data) {
        throw std::runtime_error("text::update(): unable to map index buffer for initialization.");
    }

    for (int i = 0; i < _glyph_capacity; ++i) {
        index_data[i * 6    ] = static_cast<unsigned short>(i * 4);
        index_data[i * 6 + 1] = static_cast<unsigned short>(i * 4 + 1);
        index_data[i * 6 + 2] = static_cast<unsigned short>(i * 4 + 2);
        index_data[i * 6 + 3] = static_cast<unsigned short>(i * 4);
        index_data[i * 6 + 4] = static_cast<unsigned short>(i * 4 + 2);
        index_data[i * 6 + 5] = static_cast<unsigned short>(i * 4 + 3);
    }
    device->main_context()->unmap_buffer(_index_buffer);
#endif
    update();
}

text::~text()
{
    _vertex_array.reset();
    _vertex_buffer.reset();
    _index_buffer.reset();
}

const font_face_cptr&
text::font() const
{
    return _font;
}

const font_face::style_type
text::text_style() const
{
    return _text_style;
}

const std::string&
text::text_string() const
{
    return _text_string;
}

void
text::text_string(const std::string& str)
{
    text_string(str, text_style());
}

void
text::text_string(const std::string& str,
                  const font_face::style_type stl)
{
    _text_string = str;
    _text_style  = stl;
    update();
}

bool
text::text_kerning() const
{
    return _text_kerning;
}

void
text::text_kerning(bool k)
{
    _text_kerning = k;
}

const math::vec4f&
text::text_color() const
{
    return _text_color;
}

void
text::text_color(const math::vec4f& c)
{
    _text_color = c;
}

const math::vec4f&
text::text_outline_color() const
{
    return _text_outline_color;
}

void
text::text_outline_color(const math::vec4f& c)
{
    _text_outline_color = c;
}

const math::vec4f&
text::text_shadow_color() const
{
    return _text_shadow_color;
}

void
text::text_shadow_color(const math::vec4f& c)
{
    _text_shadow_color = c;
}

const math::vec2i&
text::text_shadow_offset() const
{
    return _text_shadow_offset;
}

void
text::text_shadow_offset(const math::vec2i& o)
{
    _text_shadow_offset = o;
}

const math::vec2i&
text::text_bounding_box() const
{
    return _text_bounding_box;
}

void
text::update()
{
    if (_glyph_capacity < _text_string.size()) {
        // resize the buffers
        if (render_device_ptr device = _render_device.lock()) {
#if GEOM_SHADER_FONT == 1
            _glyph_capacity  = static_cast<int>(_text_string.size() + _text_string.size() / 2); // make it 50% bigger as required currently

            int num_vertices = _glyph_capacity; 
            if (!device->resize_buffer(_vertex_buffer, num_vertices * sizeof(vertex))) {
                err() << log::error
                      << "text::update(): unable to resize vertex buffer (size : " << num_vertices * sizeof(vertex) << ")." << log::end;
                return;
            }
#else
            _indices_count   = 0;
            _glyph_capacity  = static_cast<int>(_text_string.size() + _text_string.size() / 2); // make it 50% bigger as required currently

            int num_vertices = _glyph_capacity * 4; 
            int num_indices  = _glyph_capacity * 6;
            if (!device->resize_buffer(_vertex_buffer, num_vertices * sizeof(vertex))) {
                err() << log::error
                      << "text::update(): unable to resize vertex buffer (size : " << num_vertices * sizeof(vertex) << ")." << log::end;
                return;
            }
            if (!device->resize_buffer(_index_buffer, num_indices * sizeof(unsigned short))) {
                err() << log::error
                      << "text::update(): unable to resize index buffer (size : " << num_indices * sizeof(unsigned short) << ")." << log::end;
                return;
            }
            // fill index data
            unsigned short* index_data  = static_cast<unsigned short*>(device->main_context()->map_buffer(_index_buffer, ACCESS_WRITE_INVALIDATE_BUFFER));

            if (0 == index_data) {
                err() << log::error
                      << "text::update(): unable to map index buffer for initialization (size : " << num_indices * sizeof(unsigned short) << ")." << log::end;
                return;
            }

            for (int i = 0; i < _glyph_capacity; ++i) {
                index_data[i * 6    ] = static_cast<unsigned short>(i * 4);
                index_data[i * 6 + 1] = static_cast<unsigned short>(i * 4 + 1);
                index_data[i * 6 + 2] = static_cast<unsigned short>(i * 4 + 2);
                index_data[i * 6 + 3] = static_cast<unsigned short>(i * 4);
                index_data[i * 6 + 4] = static_cast<unsigned short>(i * 4 + 2);
                index_data[i * 6 + 5] = static_cast<unsigned short>(i * 4 + 3);
            }
            device->main_context()->unmap_buffer(_index_buffer);
#endif
        }
        else  {
            err() << log::error
                  << "text::update(): unable to optain render device from weak pointer." << log::end;
            return;
        }
    }

    if (render_context_ptr context = _render_context.lock()) {
        using namespace scm::math;

#if GEOM_SHADER_FONT == 1
        if (_text_string.empty()) {
            _indices_count     = 0;
            _text_bounding_box = math::vec2i(0, 0);
        }
        else {
            scoped_buffer_map vb_map(context, _vertex_buffer, 0, _text_string.size() * sizeof(vertex), ACCESS_WRITE_INVALIDATE_BUFFER);

            if (!vb_map) {
                err() << log::error
                      << "text::update(): unable to map vertex buffer." << log::end;
                return;
            }
            vertex*const    vertex_data = reinterpret_cast<vertex*const>(vb_map.data_ptr());
            vec2i           current_pos = vec2i(0, 0);
            int             current_lw  = 0;
            char            prev_char   = 0;

            _indices_count     = 0;
            _text_bounding_box = vec2i(0, _font->line_advance(_text_style));
            assert(_text_string.size() < (6 * (std::numeric_limits<unsigned short>::max)()));

            std::for_each(_text_string.begin(), _text_string.end(), [&](char cur_char) -> void {
                using namespace scm::gl;
                using namespace scm::math;

                if (cur_char == '\n') {
                    current_pos.x         = 0;
                    current_pos.y        -= _font->line_advance(_text_style);
                    prev_char             = 0;
                    _text_bounding_box.y += _font->line_advance(_text_style);
                    _text_bounding_box.x  = max(current_lw, _text_bounding_box.x);
                    current_lw            = 0;
                }
                else if (font_face::min_char <= cur_char && cur_char < font_face::max_char) {
                    const font_face::glyph_info& cur_glyph = _font->glyph(cur_char, _text_style);
                    // kerning
                    if (_text_kerning && prev_char) {
                        current_pos.x += _font->kerning(prev_char, cur_char, _text_style);
                    }

                    vec2f pos  = vec2f(current_pos + cur_glyph._bearing);   
                    vec2f bbox = vec2f(cur_glyph._box_size);   
                    vertex_data[_indices_count].pos_bbox = vec4f(pos, bbox.x, bbox.y);
                    vertex_data[_indices_count].tex_bbox = vec4f(cur_glyph._texture_origin, cur_glyph._texture_box_size.x, cur_glyph._texture_box_size.y);

                    _indices_count += 1;
                    // advance the position
                    current_pos.x += cur_glyph._advance;
                    current_lw    += cur_glyph._advance;

                    // remember just drawn glyph for kerning
                    prev_char = cur_char;
                }
                else {
                }
            });
            _text_bounding_box.x  = max(current_lw, _text_bounding_box.x);
        }
#else
        vec2i           current_pos = vec2i(0, 0);
        int             current_lw  = 0;
        char            prev_char   = 0;
        //vertex*         vertex_data = static_cast<vertex*>(context->map_buffer_range(_vertex_buffer, 0, 4 * _text_string.size() * sizeof(vertex), ACCESS_WRITE_INVALIDATE_BUFFER));
        vertex*         vertex_data = static_cast<vertex*>(context->map_buffer(_vertex_buffer, ACCESS_WRITE_INVALIDATE_BUFFER));

        if (0 == vertex_data) {
            err() << log::error
                  << "text::update(): unable to map vertex element or index buffer." << log::end;
            return;
        }

        _indices_count     = 0;
        _text_bounding_box = vec2i(0, _font->line_advance(_text_style));
        assert(_text_string.size() < (6 * (std::numeric_limits<unsigned short>::max)()));
        //unsigned short str_size = static_cast<unsigned short>( _text_string.size());
        size_t i = 0;
        std::for_each(_text_string.begin(), _text_string.end(), [&](char cur_char) -> void {
        //for (size_t i = 0; i < _text_string.size(); ++i) {
        //    char  cur_char = _text_string[i];

            using namespace scm::gl;
            using namespace scm::math;

            if (cur_char == '\n') {
                current_pos.x         = 0;
                current_pos.y        -= _font->line_advance(_text_style);
                prev_char             = 0;
                _text_bounding_box.y += _font->line_advance(_text_style);
                _text_bounding_box.x  = max(current_lw, _text_bounding_box.x);
                current_lw            = 0;
            }
            else {
                //if (!isalnum(cur_char) && (cur_char != ' ')) {
                //    continue;
                //}
                const font_face::glyph_info& cur_glyph = _font->glyph(cur_char, _text_style);
                // kerning
                if (_text_kerning && prev_char) {
                    current_pos.x += _font->kerning(prev_char, cur_char, _text_style);
                }

                vertex_data[i * 4    ].pos = vec2f(current_pos + cur_glyph._bearing);                                   // 00
                vertex_data[i * 4 + 1].pos = vec2f(current_pos + cur_glyph._bearing + vec2i(cur_glyph._box_size.x, 0)); // 10
                vertex_data[i * 4 + 2].pos = vec2f(current_pos + cur_glyph._bearing + cur_glyph._box_size);             // 11
                vertex_data[i * 4 + 3].pos = vec2f(current_pos + cur_glyph._bearing + vec2i(0, cur_glyph._box_size.y)); // 01

                vertex_data[i * 4    ].tex = cur_glyph._texture_origin;                                              // 00
                vertex_data[i * 4 + 1].tex = cur_glyph._texture_origin + vec2f(cur_glyph._texture_box_size.x, 0.0f); // 10
                vertex_data[i * 4 + 2].tex = cur_glyph._texture_origin + cur_glyph._texture_box_size;                // 11
                vertex_data[i * 4 + 3].tex = cur_glyph._texture_origin + vec2f(0.0f, cur_glyph._texture_box_size.y); // 01

                //vertex_data[i * 4    ].tex.z = static_cast<float>(_text_style);
                //vertex_data[i * 4 + 1].tex.z = static_cast<float>(_text_style);
                //vertex_data[i * 4 + 2].tex.z = static_cast<float>(_text_style);
                //vertex_data[i * 4 + 3].tex.z = static_cast<float>(_text_style);

                _indices_count += 6;
                ++i;
                // advance the position
                current_pos.x += cur_glyph._advance;
                current_lw    += cur_glyph._advance;

                // remember just drawn glyph for kerning
                prev_char = cur_char;
            }
        });
        _text_bounding_box.x  = max(current_lw, _text_bounding_box.x);
        
        context->unmap_buffer(_vertex_buffer);
#endif
    }
    else {
        err() << log::error
                << "text::update(): unable to optain render context from weak pointer." << log::end;
        return;
    }
}

} // namespace gl
} // namespace scm
