
#include "context_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_util/window_management/display.h>
#include <scm/gl_util/window_management/pixel_format.h>
#include <scm/gl_util/window_management/wm_win32/display_impl_win32.h>
#include <scm/gl_util/window_management/wm_win32/error_win32.h>
#include <scm/gl_util/window_management/wm_win32/pixel_format_selection_win32.h>
#include <scm/gl_util/window_management/wm_win32/wgl_extensions.h>

namespace scm {
namespace gl {
namespace wm {

context::context_impl::context_impl(const surface&  in_surface,
                                    const context&  in_share_ctx)
{
    try {
#if 0
    if (!_wgl->is_supported("WGL_ARB_create_context")){
    }
    else {
        std::vector<int>  ctx_attribs;
		if(desc.version_major() > 2) {
            ctx_attribs.push_back(WGL_CONTEXT_MAJOR_VERSION_ARB);       ctx_attribs.push_back(desc.version_major());
            ctx_attribs.push_back(WGL_CONTEXT_MINOR_VERSION_ARB);       ctx_attribs.push_back(desc.version_minor());
            int ctx_flags = 0;
            if (desc.forward_compatible()) {
                ctx_flags |= WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB;
            }
            if (desc.debug()) {
                ctx_flags |= WGL_CONTEXT_DEBUG_BIT_ARB;
            }
            if (0 != ctx_flags) {
                ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(ctx_flags);
            }
            if (_wgl->is_supported("WGL_ARB_create_context_profile")) {
                if (desc.compatibility_profile()) {
                    ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB);
                }
                else {
                    ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_CORE_PROFILE_BIT_ARB);
                }
            }
		}
        ctx_attribs.push_back(0);                                   ctx_attribs.push_back(0); // terminate list

        _context_handle.reset(_wgl->wglCreateContextAttribsARB(static_cast<HDC>(_device_handle.get()),
                                                               static_cast<HGLRC>(share_ctx.context_handle().get()),
                                                               static_cast<const int*>(&(ctx_attribs[0]))),
                              boost::bind<BOOL>(wglDeleteContext, _1));
    }

    if (!_context_handle) {
        DWORD           e = GetLastError();
        std::string     es;

        switch (e) {
            case ERROR_INVALID_VERSION_ARB: es.assign("ERROR_INVALID_VERSION_ARB");break;
            case ERROR_INVALID_PROFILE_ARB: es.assign("ERROR_INVALID_PROFILE_ARB");break;
            default: es.assign("unknown error");
        }

        glerr() << log::error
                << "context_win32::set_up(): "
                << "unable to create OpenGL context (wglCreateContextAttribsARB failed [" << es << "])"  << log::end;
        return (false);
    }

    if (_wgl->is_supported("WGL_EXT_swap_control")) {
        _swap_control_supported = true;
    }
    else {
        glerr() << log::warning
                << "context_win32::set_up(): "
                << "WGL_EXT_swap_control not supported, operating system default behavior used."  << log::end;
    }

#endif 
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

void
context::context_impl::cleanup()
{
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
