
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_WIN32_CONTEXT_IMPL_X_H_INCLUDED
#define SCM_GL_CORE_WM_WIN32_CONTEXT_IMPL_X_H_INCLUDED

#include <iosfwd>

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <X11/Xlib.h>
#include <GL/glx.h>

#include <scm/core/memory.h>

#include <scm/gl_core/window_management/context.h>

namespace scm {
namespace gl {
namespace wm {

namespace util {

class glx_extensions;

} // namespace util

struct context::context_impl
{
    context_impl(const surface_cptr&    in_surface,
                 const attribute_desc&  in_attributes,
                 const context_cptr&    in_share_ctx);
    virtual ~context_impl();

    bool                    make_current(const surface_cptr& in_surface, bool current) const;
    void                    cleanup();

    void                    print_context_informations(std::ostream& os) const;

    GLXContext              _context_handle;
    ::Display*const         _display;

    shared_ptr<util::glx_extensions>  _glx_extensions;

}; // class context_impl

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#endif // SCM_GL_CORE_WM_WIN32_CONTEXT_IMPL_X_H_INCLUDED
