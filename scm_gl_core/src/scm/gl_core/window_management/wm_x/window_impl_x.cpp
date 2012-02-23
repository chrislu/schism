
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "window_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/wm_x/display_impl_x.h>
#include <scm/gl_core/window_management/wm_x/util/framebuffer_config_selection.h>
#include <scm/gl_core/window_management/wm_x/util/glx_extensions.h>

namespace scm {
namespace gl {
namespace wm {

window::window_impl::window_impl(const display_cptr&      in_display,
                                 const ::Window           in_parent,
                                 const std::string&       in_title,
                                 const math::vec2i&       in_position,
                                 const math::vec2ui&      in_size,
                                 const format_desc&       in_sf)
  : surface::surface_impl(),
    _display(in_display->_impl->_display),
    _glx_extensions(in_display->_impl->_glx_extensions)
{
    try {
        if (channel_count(in_sf._color_format) < 3) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <xlib>: "
              << "window pixel format must contain RGB or RGBA color format "
              << "(requested: " << format_string(in_sf._color_format) << ").";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::stringstream fbc_err;
        _fb_config = util::framebuffer_config_selector::choose(in_display->_impl->_display, in_sf,
                                                               util::framebuffer_config_selector::window_surface,
                                                               _glx_extensions, fbc_err);
        if (0 == _fb_config) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <xlib>: "
              << "unable to select framebuffer config: "
              << fbc_err.str();
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        XVisualInfo* fb_visual = glXGetVisualFromFBConfig(in_display->_impl->_display, _fb_config);
        if (!fb_visual) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <xlib>: "
              << "unable to retrieve visual for framebuffer config " << _fb_config;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        XSetWindowAttributes wnd_attribs;
        Colormap             wnd_color_map = XCreateColormap(in_display->_impl->_display, RootWindow(in_display->_impl->_display, fb_visual->screen), fb_visual->visual, AllocNone);

        wnd_attribs.colormap          = wnd_color_map;
        wnd_attribs.background_pixmap = None;
        wnd_attribs.border_pixel      = 0;
        wnd_attribs.event_mask        = StructureNotifyMask;

        ::Window parent_wnd = RootWindow(in_display->_impl->_display, fb_visual->screen);
        if (0 != in_parent) {
            parent_wnd = in_parent;
        }

        _window_handle = ::XCreateWindow(in_display->_impl->_display, parent_wnd,
                                         in_position.x, in_position.y, in_size.x, in_size.y, 0,
                                         fb_visual->depth, InputOutput, fb_visual->visual,
                                         CWColormap, &wnd_attribs);
        if (!_window_handle) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <xlib>: "
              << "failed to create window.";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        _glx_window_handle = ::glXCreateWindow(in_display->_impl->_display, _fb_config, _window_handle, 0);

        if (!_glx_window_handle) {
            std::ostringstream s;
            s << "window::window_impl::window_impl() <xlib>: "
              << "failed to create glx window handle.";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        // set the drawable of the surface
        _drawable = _glx_window_handle;

        XFree(fb_visual);

        XStoreName(in_display->_impl->_display, _window_handle, in_title.c_str());
    }
    catch (...) {
        cleanup();
        throw;
    }
}

window::window_impl::~window_impl()
{
    cleanup();
}

void
window::window_impl::cleanup()
{
    if (_glx_window_handle) {
        ::glXDestroyWindow(_display, _glx_window_handle);
    }
    if (_window_handle) {
        ::XDestroyWindow(_display, _window_handle);
    }
}

void
window::window_impl::swap_buffers(int interval) const
{
    if (_glx_extensions->_swap_control_supported) {
        _glx_extensions->glXSwapIntervalSGI(interval);
    }
    ::glXSwapBuffers(_display, _glx_window_handle);
}

void
window::window_impl::show()
{
    ::XMapRaised(_display, _window_handle);
    ::XFlush(_display);
}

void
window::window_impl::hide()
{
    ::XUnmapWindow(_display, _window_handle);
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
