
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "headless_context_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <scm/core/platform/windows.h>

#include <GL/glew.h>
#include <GL/wglew.h>

#include <string>
#include <vector>

#include <boost/bind.hpp>

#include <scm/log.h>

#include <scm/gl_classic/render_context/window_context.h>

// HACK FUCKING GLEW
#define WGL_CONTEXT_PROFILE_MASK_ARB		        0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB	        0x00000001
#define WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB   0x00000002

#define ERROR_INVALID_VERSION_ARB                   0x2095
#define ERROR_INVALID_PROFILE_ARB                   0x2096

namespace scm {
namespace gl_classic {

namespace detail {

const int fixed_pbuffer_width   = 1;
const int fixed_pbuffer_height  = 1;

} // namespace detail


headless_context_win32::headless_context_win32()
{
}

headless_context_win32::~headless_context_win32()
{
}

bool
headless_context_win32::setup(const context_format& desc,
                              const window_context& partent_ctx)
{
    if (!wglewIsSupported("WGL_ARB_pbuffer")){
        scm::err() << log::error
                   << "headless_context_win32::set_up(): "
                   << "WGL_ARB_pbuffer not supported" << log::end;
        return (false);
    }
    if (partent_ctx.empty()/* != empty_context()*/) {
        scm::err() << log::error
                   << "headless_context_win32::set_up(): "
                   << "invalid parent context" << log::end;
        return (false);
    }


    std::vector<int>  pixel_desc;

    pixel_desc.push_back(WGL_ACCELERATION_ARB);     pixel_desc.push_back(WGL_FULL_ACCELERATION_ARB);
    pixel_desc.push_back(WGL_DRAW_TO_PBUFFER_ARB);  pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(WGL_SUPPORT_OPENGL_ARB);   pixel_desc.push_back(GL_TRUE);
    pixel_desc.push_back(WGL_SWAP_METHOD_ARB);      pixel_desc.push_back(WGL_SWAP_EXCHANGE_ARB);
    pixel_desc.push_back(WGL_PIXEL_TYPE_ARB);       pixel_desc.push_back(WGL_TYPE_RGBA_ARB);

    //pixel_desc.push_back(WGL_DOUBLE_BUFFER_ARB);    pixel_desc.push_back(desc.double_buffer());
    pixel_desc.push_back(WGL_STEREO_ARB);           pixel_desc.push_back(desc.stereo());
    pixel_desc.push_back(WGL_COLOR_BITS_ARB);       pixel_desc.push_back(desc.color_bits());
    pixel_desc.push_back(WGL_ALPHA_BITS_ARB);       pixel_desc.push_back(desc.alpha_bits());
    pixel_desc.push_back(WGL_DEPTH_BITS_ARB);       pixel_desc.push_back(desc.depth_bits());
    pixel_desc.push_back(WGL_STENCIL_BITS_ARB);     pixel_desc.push_back(desc.stencil_bits());
    pixel_desc.push_back(WGL_SAMPLE_BUFFERS_ARB);   pixel_desc.push_back(desc.max_samples() > 0 ? GL_TRUE : GL_FALSE);
    pixel_desc.push_back(WGL_SAMPLES_ARB);          pixel_desc.push_back(desc.max_samples());
    pixel_desc.push_back(WGL_AUX_BUFFERS_ARB);      pixel_desc.push_back(desc.max_aux_buffers());

    pixel_desc.push_back(0);                        pixel_desc.push_back(0); // terminate list

    const int         query_max_formats = 20;
    int               result_pixel_fmts[query_max_formats];
    unsigned int      result_num_pixel_fmts = 0;

    if (wglChoosePixelFormatARB(static_cast<HDC>(partent_ctx.device_handle().get()),
                                static_cast<const int*>(&(pixel_desc[0])),
                                NULL,
                                query_max_formats,
                                result_pixel_fmts,
                                &result_num_pixel_fmts) != TRUE)
    {
        scm::err() << log::error
                   << "headless_context_win32::set_up(): "
                   << "wglChoosePixelFormat failed" << log::end;
        return (false);
    }

    if (result_num_pixel_fmts < 1) {
        scm::err() << log::error
                   << "headless_context_win32::set_up(): "
                   << "wglChoosePixelFormat returned 0 matching pixel formats" << log::end;
        return (false);
    }

    _pbuffer.reset(wglCreatePbufferARB(static_cast<HDC>(partent_ctx.device_handle().get()),
                                       result_pixel_fmts[0],
                                       detail::fixed_pbuffer_width,
                                       detail::fixed_pbuffer_height,
                                       0),
                   boost::bind<BOOL>(wglDestroyPbufferARB, _1));

    if (!_pbuffer) {
        scm::err() << log::error
                   << "headless_context_win32::set_up(): "
                   << "wglCreatePbufferARB failed for format number: " << result_pixel_fmts[0] << log::end;
        return (false);
    }

    _device_handle.reset(wglGetPbufferDCARB(static_cast<HPBUFFERARB>(_pbuffer.get())),
                         boost::bind<int>(wglReleasePbufferDCARB, static_cast<HPBUFFERARB>(_pbuffer.get()), _1));

    if (!_device_handle) {
        scm::err() << log::error
                   << "headless_context_win32::set_up(): "
                   << "unable to retrive pbuffer device context (wglGetPbufferDCARB failed on pbuffer handle: "
                   << std::hex << _pbuffer.get() << ")" << log::end;
        return (false);
    }

    if (!wglewIsSupported("WGL_ARB_create_context")){
        scm::err() << log::warning
                   << "headless_context_win32::set_up(): "
                   << "WGL_ARB_create_context not supported: "
                   << "using default wglCreateContest function which does not allow request of versioned OpenGL context" << log::end;

        _context_handle.reset(wglCreateContext(static_cast<HDC>(_device_handle.get())),
                              boost::bind<BOOL>(wglDeleteContext, _1));

        if (wglShareLists(static_cast<HGLRC>(partent_ctx.context_handle().get()),
                          static_cast<HGLRC>(_context_handle.get())) == FALSE) {

            scm::err() << log::error
                       << "headless_context_win32::set_up(): "
                       << "wglShareLists failed (this: " << std::hex << _context_handle.get()
                       << ", share: " << std::hex << partent_ctx.context_handle().get() << log::end;
            return (false);
        }
    }
    else {
        std::vector<int>  ctx_attribs;
		if(desc.version_major() > 2) {
			ctx_attribs.push_back(WGL_CONTEXT_MAJOR_VERSION_ARB);       ctx_attribs.push_back(desc.version_major());
			ctx_attribs.push_back(WGL_CONTEXT_MINOR_VERSION_ARB);       ctx_attribs.push_back(desc.version_minor());
			if (desc.forward_compatible()) {
				ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB);
			}
			if (desc.debug()) {
				ctx_attribs.push_back(WGL_CONTEXT_FLAGS_ARB);           ctx_attribs.push_back(WGL_CONTEXT_DEBUG_BIT_ARB);
			}
			if (wglewIsSupported("WGL_ARB_create_context_profile")) {
				if (desc.compatibility_profile()) {
					ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB);
				}
				else {
					ctx_attribs.push_back(WGL_CONTEXT_PROFILE_MASK_ARB);    ctx_attribs.push_back(WGL_CONTEXT_CORE_PROFILE_BIT_ARB);
				}
			}
		}
        ctx_attribs.push_back(0);                                   ctx_attribs.push_back(0); // terminate list

        _context_handle.reset(wglCreateContextAttribsARB(static_cast<HDC>(_device_handle.get()),
                                                         static_cast<HGLRC>(partent_ctx.context_handle().get()),
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

        scm::err() << log::error
                   << "headless_context_win32::set_up(): "
                   << "unable to create OpenGL context (wglCreateContextAttribsARB failed [" << es << "])"  << log::end;
        return (false);
    }

    _context_format = desc;

    return (true);
}

void
headless_context_win32::cleanup()
{
    make_current(false);
    _context_handle.reset();
    _device_handle.reset();
    _pbuffer.reset();

}

bool
headless_context_win32::make_current(bool current) const
{
    return (wglMakeCurrent(static_cast<HDC>(_device_handle.get()),
                           current ? static_cast<HGLRC>(_context_handle.get()) : NULL) == TRUE ? true : false);
}

} // namespace gl_classic
} // namespace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
