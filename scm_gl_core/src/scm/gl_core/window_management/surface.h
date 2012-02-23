
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_SURFACE_H_INCLUDED
#define SCM_GL_CORE_WM_SURFACE_H_INCLUDED

#include <ostream>

#include <scm/core/memory.h>

#include <scm/gl_core/data_formats.h>

#include <scm/gl_core/window_management/wm_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace wm {

class __scm_export(gl_core) surface
{
public:
    struct __scm_export(gl_core) format_desc
    {
        data_format             _color_format;
        data_format             _depth_stencil_format;

        bool                    _double_buffer;
        bool                    _quad_buffer_stereo;

        format_desc(data_format color_fmt, data_format depth_stencil_fmt,
                    bool double_buffer, bool quad_buffer_stereo = false);

        friend __scm_export(gl_core) std::ostream& operator<<(std::ostream& out_stream, const format_desc& pf);
    }; // class format_desc

public:
    virtual ~surface();

    const display_cptr&         associated_display() const;
    const format_desc&          surface_format() const;

    static const format_desc&   default_format();

protected:
    display_cptr                _associated_display;
    format_desc                 _format;

protected:
    struct surface_impl;
    shared_ptr<surface_impl>    _impl;

protected: // will be public again when this is an abstract class
    surface(const display_cptr& in_display,
            const format_desc&  in_sf);

private:
    // non_copyable
    surface(const surface&);
    surface& operator=(const surface&);

    friend class scm::gl::wm::context;
    friend class scm::gl::wm::window;
    friend class scm::gl::wm::headless_surface;
}; // class surface

} // namespace wm
} // namepspace gl
} // namepspace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_WM_SURFACE_H_INCLUDED
