
#include "text_box.h"

#include <vector>

#include <boost/algorithm/string.hpp>

#include <scm/core/utilities/foreach.h>
#include <scm/core/gui/font_renderer.h>

using namespace scm;
using namespace scm::gui;

text_box::text_box()
  : _orientation(orient_horizontal),
    _flow(flow_top_to_bottom),
    _hor_alignment(hor_align_left),
    _vert_alignment(vert_align_top)
{
    text_box::line          new_line;
    new_line._alignment     = hor_alignment();

    // insert first line
    _lines.push_front(new_line);
}

text_box::~text_box()
{
}

void text_box::orientation(text_orientation ori)
{
    _orientation = ori;
}

void text_box::flow(text_flow flow)
{
    _flow = flow;
}

void text_box::hor_alignment(text_hor_alignment align)
{
    _hor_alignment = align;
}

void text_box::vert_alignment(text_vert_alignment align)
{
    _vert_alignment = align;
}

text_orientation text_box::orientation() const
{
    return (_orientation);
}

text_flow text_box::flow() const
{
    return (_flow);
}

text_hor_alignment text_box::hor_alignment() const
{
    return (_hor_alignment);
}

text_vert_alignment text_box::vert_alignment() const
{
    return (_vert_alignment);
}

void text_box::font(const font::face_ptr& ptr)
{
    _font_renderer->active_font(ptr);
}


const font::face_ptr& text_box::font() const
{
    return (_font_renderer->active_font());
}

void text_box::append_string(const std::string&      txt,
                             bool                    unl,
                             font::face::style_type  stl)
{
    append_string(txt, math::vec4f_t(1.f), unl, stl);
}

void text_box::append_string(const std::string&      txt,
                             const math::vec3f_t     col,
                             bool                    unl,
                             font::face::style_type  stl)
{
    append_string(txt, math::vec4f_t(col, 1.f), unl, stl);
}

void text_box::append_string(const std::string&      txt,
                             const math::vec4f_t     col,
                             bool                    unl,
                             font::face::style_type  stl)
{
    using namespace std;
    using namespace boost::algorithm;

    typedef vector<string>      string_vec;

    string_vec  new_lines;

    split(new_lines, txt, is_any_of("\n"));

    text_box::line_fragment new_frag;
    new_frag._color     = col;
    new_frag._style     = stl;
    new_frag._underline = unl;

    foreach (const std::string& l, new_lines) {
        new_frag._text.assign(l);
        _lines.front()._fragments.push_back(new_frag);
        _lines.front()._alignment = hor_alignment();

        if (!l.empty()) {
            _lines.push_front(text_box::line());
        }
    }
}

void text_box::draw_text() const
{
    // how many lines are possible
    unsigned line_advance   = _font_renderer->get_current_line_advance();
    std::size_t max_lines   = math::max(0, _size.y - _content_margins.z - _content_margins.w) / line_advance;

    // calculate starting position
    int   vert_start_pos;

    if (_flow == flow_top_to_bottom) {
        // top_left_pos
        vert_start_pos = _position.y + _size.y - _content_margins.w;
        vert_start_pos -= math::min(max_lines, _lines.size()) * line_advance;
    }
    else { // _flow == flow_bottom_to_top
        // low_left_pos
        vert_start_pos = _position.y + _content_margins.y;
    }

    math::vec2i_t   start_pos;
    unsigned        num_drawn_lines = 0;

    foreach (const text_box::line& l, _lines) {

        // calculate line length
        unsigned line_len = 0;

        foreach (const text_box::line_fragment& lf, l._fragments) {
            line_len += _font_renderer->calculate_text_width(lf._text, lf._style);
        }

        if (l._alignment == hor_align_left) {
            start_pos = math::vec2i_t(_position.x + _content_margins.x, vert_start_pos);
        }
        else if (l._alignment == hor_align_right) {
            start_pos = math::vec2i_t(_position.x + _size.x - _content_margins.y - line_len, vert_start_pos);
        }
        else if (l._alignment == hor_align_center) {
            start_pos = math::vec2i_t((_position.x + _size.x - _content_margins.y) / 2 - line_len / 2, vert_start_pos);
        }

        foreach (const text_box::line_fragment& lf, l._fragments) {
            unsigned frag_len = _font_renderer->calculate_text_width(lf._text, lf._style);

            _font_renderer->draw_string(start_pos,
                                        lf._text,
                                        lf._color,
                                        lf._underline,
                                        lf._style);

            start_pos.x += frag_len;
        }
        // advance line pos
        vert_start_pos     += line_advance;
        num_drawn_lines    += 1;

        if (num_drawn_lines >= max_lines) {
            break;
        }
        // break if cur_line > max_lines
    }
}
