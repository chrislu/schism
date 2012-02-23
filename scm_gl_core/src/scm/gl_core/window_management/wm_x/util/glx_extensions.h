
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_WM_X_GLX_EXTENSIONS_H_INCLUDED
#define SCM_GL_CORE_WM_X_GLX_EXTENSIONS_H_INCLUDED

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <ostream>
#include <set>
#include <string>

#include <X11/Xlib.h>
#include <GL/glx.h>
#include <scm/gl_core/window_management/GL/glxext.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {

class glx_extensions
{
private:
    typedef std::set<std::string> string_set;

public:

//    // GLX 1.3
//    PFNGLXGETFBCONFIGSPROC              glXGetFBConfigs;
//    PFNGLXCHOOSEFBCONFIGPROC            glXChooseFBConfig;
//    PFNGLXGETFBCONFIGATTRIBPROC         glXGetFBConfigAttrib;
//    PFNGLXGETVISUALFROMFBCONFIGPROC     glXGetVisualFromFBConfig;
//    PFNGLXCREATEWINDOWPROC              glXCreateWindow;
//    PFNGLXDESTROYWINDOWPROC             glXDestroyWindow;
//    PFNGLXCREATEPIXMAPPROC              glXCreatePixmap;
//    PFNGLXDESTROYPIXMAPPROC             glXDestroyPixmap;
//    PFNGLXCREATEPBUFFERPROC             glXCreatePbuffer;
//    PFNGLXDESTROYPBUFFERPROC            glXDestroyPbuffer;
//    PFNGLXQUERYDRAWABLEPROC             glXQueryDrawable;
//    PFNGLXCREATENEWCONTEXTPROC          glXCreateNewContext;
//    PFNGLXMAKECONTEXTCURRENTPROC        glXMakeContextCurrent;
//    PFNGLXGETCURRENTREADDRAWABLEPROC    glXGetCurrentReadDrawable;
//    PFNGLXGETCURRENTDISPLAYPROC         glXGetCurrentDisplay;
//    PFNGLXQUERYCONTEXTPROC              glXQueryContext;
//    PFNGLXSELECTEVENTPROC               glXSelectEvent;
//    PFNGLXGETSELECTEDEVENTPROC          glXGetSelectedEvent;
//
//    // GLX 1.4
//    PFNGLXGETPROCADDRESSPROC            glXGetProcAddress;
//
//    // GLX_ARB_get_proc_address
//    PFNGLXGETPROCADDRESSARBPROC         glXGetProcAddressARB;

    // GLX_ARB_create_context
    PFNGLXCREATECONTEXTATTRIBSARBPROC   glXCreateContextAttribsARB;

    // GLX_SGI_swap_control
    PFNGLXSWAPINTERVALSGIPROC           glXSwapIntervalSGI;

    bool                                _swap_control_supported;

public:
    glx_extensions();

    bool            initialize(Display*const in_display, std::ostream& os);
    bool            is_initialized() const;
    bool            is_supported(const std::string& ext) const;

private:
    string_set      _glx_extensions;
    bool            _initialized;

}; // class glx_extensions

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm


#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
#endif // SCM_GL_CORE_WM_X_GLX_EXTENSIONS_H_INCLUDED
