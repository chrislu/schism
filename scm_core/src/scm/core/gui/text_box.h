
#ifndef SCM_GUI_TEXT_BOX_H_INCLUDED
#define SCM_GUI_TEXT_BOX_H_INCLUDED

#include <list>
#include <string>

#include <boost/scoped_ptr.hpp>

#include <scm/core/font/face.h>
#include <scm/core/gui/gui.h>
#include <scm/core/gui/frame.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gui {

class font_renderer;

class __scm_export(core) text_box : public frame
{
protected:
    typedef struct {
        std::string                 _text;
        scm::math::vec4f            _color;
        font::face::style_type      _style;
        bool                        _underline;
    } line_fragment;
    typedef struct {
        std::list<line_fragment>    _fragments;
        text_hor_alignment          _alignment;
    } line;

public:
    text_box();
    virtual ~text_box();

    void                        orientation(text_orientation        /*ori*/);
    text_orientation            orientation() const;

    void                        flow(text_flow                      /*flow*/);
    text_flow                   flow() const;

    void                        hor_alignment(text_hor_alignment    /*align*/);
    text_hor_alignment          hor_alignment() const;

    void                        vert_alignment(text_vert_alignment  /*align*/);
    text_vert_alignment         vert_alignment() const;

    void                        font(const font::face_ptr& /*ptr*/);
    const font::face_ptr&       font() const;


    void                        append_string(const std::string&      /*txt*/,
                                              bool                    /*unl*/ = false,
                                              font::face::style_type  /*stl*/ = font::face::regular);
    void                        append_string(const std::string&      /*txt*/,
                                              const scm::math::vec3f  /*col*/,
                                              bool                    /*unl*/ = false,
                                              font::face::style_type  /*stl*/ = font::face::regular);
    void                        append_string(const std::string&      /*txt*/,
                                              const scm::math::vec4f  /*col*/,
                                              bool                    /*unl*/ = false,
                                              font::face::style_type  /*stl*/ = font::face::regular);

protected:
    text_orientation            _orientation;
    text_flow                   _flow;
    text_hor_alignment          _hor_alignment;
    text_vert_alignment         _vert_alignment;

    boost::scoped_ptr<font_renderer> _font_renderer;

    std::list<line>             _lines;


    void                        draw_text() const;

private:

}; // class text_box

} // namespace gui
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GUI_TEXT_BOX_H_INCLUDED
