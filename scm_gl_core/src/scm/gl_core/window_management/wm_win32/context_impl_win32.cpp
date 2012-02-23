
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "context_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/surface.h>
#include <scm/gl_core/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_core/window_management/wm_win32/surface_impl_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/error_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/pixel_format_selection_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/wgl_extensions.h>

namespace scm {
namespace gl {
namespace wm {

context::context_impl::context_impl(const surface_cptr&    in_surface,
                                    const attribute_desc&  in_attributes,
                                    const context_cptr&    in_share_ctx)
  : _context_handle(0),
    _swap_control_supported(false),
    _wgl_extensions(in_surface->associated_display()->_impl->_wgl_extensions)
{
    try {
        if (   !_wgl_extensions->is_supported("WGL_ARB_create_context")
            || !_wgl_extensions->is_supported("WGL_ARB_create_context_profile")) {
            std::ostringstream s;
            s << "context::context_impl::context_impl() <win32>: " 
              << "missing WGL extensions (WGL_ARB_create_context, WGL_ARB_create_context_profile)";
            throw(std::runtime_error(s.str()));
        }
        std::vector<int>  ctx_attribs;

        if (!in_attributes._es_profile) {
            if(in_attributes._version_major > 2) {
                ctx_attribs.push_back(WGL_CONTEXT_MAJOR_VERSION_ARB);       ctx_attribs.push_back(in_attributes._version_major);
                ctx_attribs.push_back(WGL_CONTEXT_MINOR_VERSION_ARB);       ctx_attribs.push_back(in_attributes._version_minor);
                int ctx_flags = 0;
                if (in_attributes._forward_compatible) {
                    ctx_flags |= WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB;
                }
                if (in_attributes._debug) {
                    ctx_flags |= WGL_CONTEXT_DEBUG_BIT_ARB;
                }
                if (0 != ctx_flags) {
                    ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(ctx_flags);
                }
                if (in_attributes._compatibility_profile) {
                    ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB);
                }
                else {
                    ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_CORE_PROFILE_BIT_ARB);
                }
            }
        }
        else {
            ctx_attribs.push_back(WGL_CONTEXT_MAJOR_VERSION_ARB);       ctx_attribs.push_back(in_attributes._version_major);
            ctx_attribs.push_back(WGL_CONTEXT_MINOR_VERSION_ARB);       ctx_attribs.push_back(in_attributes._version_minor);
            ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);               ctx_attribs.push_back(WGL_CONTEXT_ES2_PROFILE_BIT_EXT);
        }
        ctx_attribs.push_back(0);                                   ctx_attribs.push_back(0); // terminate list

        HGLRC share_ctx = 0;
        if (in_share_ctx) {
            share_ctx = in_share_ctx->_impl->_context_handle;
        }

        _context_handle = _wgl_extensions->wglCreateContextAttribsARB(in_surface->_impl->_device_handle,
                                                                      share_ctx,
                                                                      static_cast<const int*>(&(ctx_attribs[0])));

        if (!_context_handle) {

            DWORD           e = GetLastError();
            std::string     es;

            switch (e) {
                case ERROR_INVALID_VERSION_ARB:  es.assign("ERROR_INVALID_VERSION_ARB");break;
                case ERROR_INVALID_PROFILE_ARB:  es.assign("ERROR_INVALID_PROFILE_ARB");break;
                case ERROR_INVALID_OPERATION:    es.assign("ERROR_INVALID_OPERATION");break;
                case ERROR_DC_NOT_FOUND:         es.assign("ERROR_DC_NOT_FOUND");break;
                case ERROR_INVALID_PIXEL_FORMAT: es.assign("ERROR_INVALID_PIXEL_FORMAT");break;
                case ERROR_NO_SYSTEM_RESOURCES:  es.assign("ERROR_NO_SYSTEM_RESOURCES");break;
                case ERROR_INVALID_PARAMETER:    es.assign("ERROR_INVALID_PARAMETER");break;
                default: es.assign("unknown error");
            }

            //char* error_msg;

            //FormatMessage(  FORMAT_MESSAGE_IGNORE_INSERTS
            //              | FORMAT_MESSAGE_FROM_SYSTEM
            //              | FORMAT_MESSAGE_ALLOCATE_BUFFER,
            //              0,
            //              e,
            //              MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            //              (LPTSTR)&error_msg,
            //              1024,
            //              0);

            //LocalFree(error_msg);

            std::ostringstream s;
            s << "context::context_impl::context_impl() <win32>: " 
              << "unable to create OpenGL context (wglCreateContextAttribsARB failed) "
              << "(system message: " << es/*util::win32_error_message()*/ << ")";
            throw(std::runtime_error(s.str()));
        }
        else {
            // test code
            //if (in_share_ctx) {
            //    if (TRUE != wglShareLists(in_share_ctx->_impl->_context_handle, _context_handle)) {
            //        std::ostringstream s;
            //        s << "context::context_impl::context_impl() <win32>: " 
            //          << "unable to create OpenGL context share lists (wglShareLists failed).";
            //        throw std::runtime_error(s.str());
            //    }
            //}
        }

        if (_wgl_extensions->is_supported("WGL_EXT_swap_control")) {
            _swap_control_supported = true;
        }
        else {
            err() << log::warning
                  << "context::context_impl::context_impl() <win32>: "
                  << "WGL_EXT_swap_control not supported, operating system default behavior used."  << log::end;
        }
    }
    catch(...) {
        cleanup();
        throw;
    }
}

context::context_impl::~context_impl()
{
    cleanup();
}

bool
context::context_impl::make_current(const surface_cptr& in_surface, bool current) const
{
    return ((TRUE == ::wglMakeCurrent(in_surface->_impl->_device_handle, current ? _context_handle : 0)) ? true : false);
}

void
context::context_impl::cleanup()
{
    if (_context_handle) {
        if (FALSE == ::wglDeleteContext(_context_handle)) {
            err() << log::error
                  << "context::context_impl::~context_impl() <win32>: " 
                  << "unable to destroy gl context handle "
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
        }
    }
}

void
context::context_impl::print_context_informations(std::ostream& os) const
{
    os << "OpenGL WGL render device" << std::endl;
    os << *_wgl_extensions;
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
