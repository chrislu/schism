
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "framebuffer_config_selection.h"

#include <scm/core/platform/platform.h>

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <sstream>
#include <string>
#include <vector>

#include <scm/gl_core/constants.h>

#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/wm_x/display_impl_x.h>
#include <scm/gl_core/window_management/wm_x/util/glx_extensions.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {

/*static*/
GLXFBConfig
framebuffer_config_selector::choose(::Display*const                   in_display,
                                    const surface::format_desc&       in_pfd,
                                    const surface_type                in_surface_type,
                                    const shared_ptr<glx_extensions>& in_glx,
                                    std::ostream&                     out_stream)
{
    std::vector<int> visual_attribs;

    visual_attribs.push_back(GLX_X_RENDERABLE);
    visual_attribs.push_back(True);

    if (in_surface_type == window_surface) {
        visual_attribs.push_back(GLX_DRAWABLE_TYPE);
        visual_attribs.push_back(GLX_WINDOW_BIT);
    }
    else if (in_surface_type == pbuffer_surface) {
        visual_attribs.push_back(GLX_DRAWABLE_TYPE);
        visual_attribs.push_back(GLX_PBUFFER_BIT);
    }
    else {
        out_stream << "framebuffer_config_selector::choose() <xlib>: "
                   << "unknown surface type.";
        return (0);
    }

    visual_attribs.push_back(GLX_RENDER_TYPE);
    visual_attribs.push_back(GLX_RGBA_BIT);

    visual_attribs.push_back(GLX_X_VISUAL_TYPE);
    visual_attribs.push_back(GLX_TRUE_COLOR);

    visual_attribs.push_back(GLX_RED_SIZE);
    visual_attribs.push_back(size_of_channel(in_pfd._color_format) * 8);

    visual_attribs.push_back(GLX_GREEN_SIZE);
    visual_attribs.push_back(size_of_channel(in_pfd._color_format) * 8);

    visual_attribs.push_back(GLX_BLUE_SIZE);
    visual_attribs.push_back(size_of_channel(in_pfd._color_format) * 8);

    visual_attribs.push_back(GLX_ALPHA_SIZE);
    if ((channel_count(in_pfd._color_format) > 3)) {
        visual_attribs.push_back(size_of_channel(in_pfd._color_format) * 8);
    }
    else {
        visual_attribs.push_back(0);
    }

    visual_attribs.push_back(GLX_DEPTH_SIZE);
    visual_attribs.push_back(size_of_depth_component(in_pfd._depth_stencil_format) * 8);

    visual_attribs.push_back(GLX_STENCIL_SIZE);
    visual_attribs.push_back(size_of_stencil_component(in_pfd._depth_stencil_format) * 8);

    visual_attribs.push_back(GLX_DOUBLEBUFFER);
    visual_attribs.push_back(in_pfd._double_buffer);

    visual_attribs.push_back(GLX_STEREO);
    visual_attribs.push_back(in_pfd._quad_buffer_stereo);

    visual_attribs.push_back(GLX_SAMPLE_BUFFERS);
    visual_attribs.push_back(0);

    visual_attribs.push_back(GLX_SAMPLES);
    visual_attribs.push_back(0);

    visual_attribs.push_back(GLX_AUX_BUFFERS);
    visual_attribs.push_back(0);

    // terminate list
    visual_attribs.push_back(0);
    visual_attribs.push_back(0);

    int glx_major = 0;
    int glx_minor = 0;

    // FBConfigs were added in GLX version 1.3
    if (   !::glXQueryVersion(in_display, &glx_major, &glx_minor)
        || ((glx_major == 1) && (glx_minor < 3)) || (glx_major < 1)) {
        out_stream << "framebuffer_config_selector::choose() <xlib>: "
                   << "invalid GLX version - 1.3 required"
                   << "(GLX version: " << glx_major << "." << glx_major << ").";
        return (0);
    }

    int fb_count = 0;
    GLXFBConfig *fb_configs = ::glXChooseFBConfig(in_display, ::XDefaultScreen(in_display),
                                                  static_cast<const int*>(&(visual_attribs[0])), &fb_count);
    if (!fb_configs) {
        std::ostringstream s;
        out_stream << "framebuffer_config_selector::choose() <xlib>: "
                   << "unable to receive framebuffer configs - requested pixel format: " << std::endl
                   << in_pfd;
        return (0);
    }

    GLXFBConfig sel_fb_config = fb_configs[0];
    XFree(fb_configs);

    return (sel_fb_config);
}

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
