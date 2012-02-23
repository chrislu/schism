
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "headless_surface_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/window.h>
#include <scm/gl_core/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_core/window_management/wm_win32/window_impl_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/error_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/pixel_format_selection_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/wgl_extensions.h>

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
    _wgl_extensions(in_parent_wnd->associated_display()->_impl->_wgl_extensions)
{
    try {
        const surface::format_desc&         pfd  = in_parent_wnd->surface_format();
        const display_cptr&                 disp = in_parent_wnd->associated_display();

        if (!_wgl_extensions) {
            std::ostringstream s;
            s << "headless_surface::headless_surface_impl::headless_surface_impl() <win32>: "
              << "unable to get wgl extensions from parent window.";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::stringstream pfd_err;
        int pfd_num = util::pixel_format_selector::choose(disp->_impl->_device_handle,
                                                          pfd, util::pixel_format_selector::pbuffer_surface,
                                                          _wgl_extensions, pfd_err);
        if (0 == pfd_num) {
            std::ostringstream s;
            s << "headless_surface::headless_surface_impl::headless_surface_impl() <win32>: "
              << "unable select pixel format: "
              << pfd_err.str();
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        _pbuffer_handle = _wgl_extensions->wglCreatePbufferARB(in_parent_wnd->_impl->_device_handle,
                                                 pfd_num,
                                                 detail::fixed_pbuffer_width,
                                                 detail::fixed_pbuffer_height,
                                                 0);
        if (!_pbuffer_handle) {
            std::ostringstream s;
            s << "headless_surface::headless_surface_impl::headless_surface_impl() <win32>: "
              << "unable to create pbuffer, wglCreatePbufferARB failed for format number: " << pfd_num;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        _device_handle = _wgl_extensions->wglGetPbufferDCARB(_pbuffer_handle);

        if (!_device_handle) {
            std::ostringstream s;
            s << "headless_surface::headless_surface_impl::headless_surface_impl() <win32>: "
              << "unable to retrive pbuffer device context (wglGetPbufferDCARB failed on pbuffer handle: "
              << std::hex << _pbuffer_handle;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }
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
    if (_device_handle) {
        if (0 == _wgl_extensions->wglReleasePbufferDCARB(_pbuffer_handle, _device_handle)) {
            err() << log::error
                  << "headless_surface::headless_surface_impl::cleanup() <win32>: " 
                  << "unable to release pbuffer device context"
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
        }
    }
    if (_pbuffer_handle) {
        if (FALSE == _wgl_extensions->wglDestroyPbufferARB(_pbuffer_handle)) {
            err() << log::error
                  << "headless_surface::headless_surface_impl::cleanup() <win32>: " 
                  << "unable to destroy pbuffer"
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
        }
    }
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
