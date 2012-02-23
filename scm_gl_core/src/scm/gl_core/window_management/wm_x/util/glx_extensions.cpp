
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "glx_extensions.h"

#if SCM_PLATFORM == SCM_PLATFORM_LINUX

#include <GL/glx.h>

#include <string>

#include <boost/tokenizer.hpp>

#include <scm/core/memory.h>

namespace scm {
namespace gl {
namespace wm {
namespace util {

glx_extensions::glx_extensions()
{
    _initialized = false;

    glXCreateContextAttribsARB      = 0;

    glXSwapIntervalSGI              = 0;

    _swap_control_supported         = false;
}

bool
glx_extensions::initialize(Display*const in_display, std::ostream& os)
{
    if (is_initialized()) {
        return (true);
    }
    int glx_major = 0;
    int glx_minor = 0;

    // FBConfigs were added in GLX version 1.3
    if (   !::glXQueryVersion(in_display, &glx_major, &glx_minor)
        || ((glx_major == 1) && (glx_minor < 4)) || (glx_major < 1)) {
        os << "glx_extensions::initialize() <xlib>: "
           << "invalid GLX version - at least 1.4 required "
           << "(GLX version: " << glx_major << "." << glx_major << ").";
        //err() << log::fatal << s.str() << log::end;
        return (false);
    }

    std::string glx_ext_string = reinterpret_cast<const char*>(glXQueryExtensionsString(in_display, XDefaultScreen(in_display)));

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> space_separator(" ");
    tokenizer                   extension_strings(glx_ext_string, space_separator);

    for (tokenizer::const_iterator i = extension_strings.begin(); i != extension_strings.end(); ++i) {
        _glx_extensions.insert(std::string(*i));
    }

    // GLX_ARB_create_context
    glXCreateContextAttribsARB      = (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((GLubyte*)"glXCreateContextAttribsARB");
    if (!glXCreateContextAttribsARB) {
        os << "glx_extensions::initialize() <xlib>: "
           << "glXCreateContextAttribsARB not supported.";
        //err() << log::fatal << s.str() << log::end;
        return (false);
    }

    // GLX_SGI_swap_control
    glXSwapIntervalSGI              = (PFNGLXSWAPINTERVALSGIPROC)glXGetProcAddress((GLubyte*)"glXSwapIntervalSGI");
    if (!glXSwapIntervalSGI) {
        os << "glx_extensions::initialize() <xlib>: "
           << "glXSwapIntervalSGI not supported.";
        //err() << log::fatal << s.str() << log::end;
    }
    else {
        _swap_control_supported = true;
    }

    _initialized = true;

    return (true);
}

bool
glx_extensions::is_initialized() const
{
    return (_initialized);
}

bool
glx_extensions::is_supported(const std::string& ext) const
{
    if (_glx_extensions.find(ext) != _glx_extensions.end()) {
        return (true);
    }
    else {
        return (false);
    }
}

} // namespace util
} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_LINUX
