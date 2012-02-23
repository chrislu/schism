
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "context_impl_x.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <exception>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <scm/log.h>
#include <scm/gl_core/constants.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/surface.h>
#include <scm/gl_core/window_management/GL/glxext.h>
#include <scm/gl_core/window_management/wm_x/display_impl_x.h>
#include <scm/gl_core/window_management/wm_x/surface_impl_x.h>
#include <scm/gl_core/window_management/wm_x/util/glx_extensions.h>

#ifndef GLX_ARB_create_context
#define GLX_CONTEXT_DEBUG_BIT_ARB          0x00000001
#define GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB 0x00000002
#define GLX_CONTEXT_MAJOR_VERSION_ARB      0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB      0x2092
#define GLX_CONTEXT_FLAGS_ARB              0x2094
#endif

#ifndef GLX_ARB_create_context_profile
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB   0x00000001
#define GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB 0x00000002
#define GLX_CONTEXT_PROFILE_MASK_ARB       0x9126
#endif


namespace scm {
namespace gl {
namespace wm {

context::context_impl::context_impl(const surface_cptr&    in_surface,
                                    const attribute_desc&  in_attributes,
                                    const context_cptr&    in_share_ctx)
  : _context_handle(0),
    _display(in_surface->associated_display()->_impl->_display),
    _glx_extensions(in_surface->associated_display()->_impl->_glx_extensions)
{
    try {
        if (   !_glx_extensions->is_supported("GLX_ARB_create_context")
            || !_glx_extensions->is_supported("GLX_ARB_create_context_profile")) {
            std::ostringstream s;
            s << "context::context_impl::context_impl() <xlib>: "
              << "missing WGL extensions (GLX_ARB_create_context, GLX_ARB_create_context_profile)";
            throw(std::runtime_error(s.str()));
        }

        std::vector<int>  ctx_attribs;

        if(in_attributes._version_major > 2) {
            ctx_attribs.push_back(GLX_CONTEXT_MAJOR_VERSION_ARB);       ctx_attribs.push_back(in_attributes._version_major);
            ctx_attribs.push_back(GLX_CONTEXT_MINOR_VERSION_ARB);       ctx_attribs.push_back(in_attributes._version_minor);
            int ctx_flags = 0;
            if (in_attributes._forward_compatible) {
                ctx_flags |= GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB;
            }
            if (in_attributes._debug) {
                ctx_flags |= GLX_CONTEXT_DEBUG_BIT_ARB;
            }
            if (0 != ctx_flags) {
                ctx_attribs.push_back(GLX_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(ctx_flags);
            }
            if (in_attributes._compatibility_profile) {
                ctx_attribs.push_back(GLX_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB);
            }
            else {
                ctx_attribs.push_back(GLX_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(GLX_CONTEXT_CORE_PROFILE_BIT_ARB);
            }
        }
        ctx_attribs.push_back(0);                                   ctx_attribs.push_back(0); // terminate list


        GLXContext share_ctx = 0;
        if (in_share_ctx) {
            share_ctx = in_share_ctx->_impl->_context_handle;
        }

        _context_handle = _glx_extensions->glXCreateContextAttribsARB(in_surface->associated_display()->_impl->_display,
                                                                      in_surface->_impl->_fb_config,
                                                                      share_ctx,
                                                                      true,
                                                                      static_cast<const int*>(&(ctx_attribs[0])));
        if (!_context_handle) {
            std::ostringstream s;
            s << "context::context_impl::context_impl() <xlib>: "
              << "unable to create OpenGL context (glXCreateContextAttribsARB failed).";
            throw(std::runtime_error(s.str()));
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
    GLXDrawable cur_drawable = (current ? in_surface->_impl->_drawable : 0);

    return (glXMakeContextCurrent(_display, cur_drawable, cur_drawable, _context_handle));
}

void
context::context_impl::cleanup()
{
    if (_context_handle) {
        ::glXDestroyContext(_display, _context_handle);
    }
}

void
context::context_impl::print_context_informations(std::ostream& os) const
{
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
