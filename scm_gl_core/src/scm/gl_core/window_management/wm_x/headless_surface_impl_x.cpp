
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "headless_surface_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/window.h>
#include <scm/gl_core/window_management/wm_x/display_impl_x.h>
#include <scm/gl_core/window_management/wm_x/window_impl_x.h>
#include <scm/gl_core/window_management/wm_x/util/framebuffer_config_selection.h>
#include <scm/gl_core/window_management/wm_x/util/glx_extensions.h>

namespace scm {
namespace gl {
namespace wm {

namespace detail {

const int fixed_pbuffer_width   = 1;
const int fixed_pbuffer_height  = 1;

} // namespace detail

headless_surface::headless_surface_impl::headless_surface_impl(const window_cptr& in_parent_wnd)
  : surface::surface_impl(),
    _pbuffer_handle(0),
    _display(in_parent_wnd->associated_display()->_impl->_display),
    _glx_extensions(in_parent_wnd->associated_display()->_impl->_glx_extensions)
{
    try {
        const surface::format_desc&         pfd  = in_parent_wnd->surface_format();
        const display_cptr&                 disp = in_parent_wnd->associated_display();

        if (!_glx_extensions) {
            std::ostringstream s;
            s << "headless_surface::headless_surface_impl::headless_surface_impl() <xlib>: "
              << "unable to get glx extensions from parent window.";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::stringstream fbc_err;
        _fb_config = util::framebuffer_config_selector::choose(disp->_impl->_display, pfd,
                                                               util::framebuffer_config_selector::pbuffer_surface,
                                                               _glx_extensions, fbc_err);
        if (0 == _fb_config) {
            std::ostringstream s;
            s << "headless_surface::headless_surface_impl::headless_surface_impl() <xlib>: "
              << "unable to select framebuffer config: "
              << fbc_err.str();
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::vector<int>    pb_attribs;

        pb_attribs.push_back(GLX_PBUFFER_WIDTH);
        pb_attribs.push_back(detail::fixed_pbuffer_width);
        pb_attribs.push_back(GLX_PBUFFER_HEIGHT);
        pb_attribs.push_back(detail::fixed_pbuffer_height);
        pb_attribs.push_back(0);
        pb_attribs.push_back(0);

        _pbuffer_handle = glXCreatePbuffer(disp->_impl->_display,
                                           _fb_config,
                                           static_cast<const int*>(&(pb_attribs[0])));
        if (!_pbuffer_handle) {
            std::ostringstream s;
            s << "headless_surface::headless_surface_impl::headless_surface_impl() <xlib>: "
              << "unable to create pbuffer, glXCreateGLXPbuffer failed for format: " << _fb_config;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        // set the drawable of the surface
        _drawable = _pbuffer_handle;
    }
    catch(...) {
        cleanup();
        throw;
    }
}

headless_surface::headless_surface_impl::~headless_surface_impl()
{
    cleanup();
}

void
headless_surface::headless_surface_impl::cleanup()
{
    if (_pbuffer_handle) {
        glXDestroyPbuffer(_display, _pbuffer_handle);
    }
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
