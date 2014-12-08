
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "gl_core.h"

#include <ostream>

#include <boost/tokenizer.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>

#include <scm/core/platform/platform.h>

#if   SCM_PLATFORM == SCM_PLATFORM_WINDOWS
#   include <scm/core/platform/windows.h>
#elif SCM_PLATFORM == SCM_PLATFORM_LINUX
#   include <GL/glx.h>
#else
#   error "unsupported platform"
#endif // SCM_PLATFORM

#include <scm/core/log/logger_state.h>

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>

namespace  {

inline
void*
sys_get_proc_address(const char *name)
{
#if   SCM_PLATFORM == SCM_PLATFORM_WINDOWS
    void* p = ::wglGetProcAddress(name);
    if (0 != p) {
        return (p);
    }
    else {
        return (::GetProcAddress(::GetModuleHandle("OpenGL32"), name));
    }
#elif SCM_PLATFORM == SCM_PLATFORM_LINUX
    return (void*) (*glXGetProcAddressARB((const GLubyte*) name));
#endif
}

template <typename retType>
inline
retType
gl_proc_address(const char *name)
{
    return (reinterpret_cast<retType>(sys_get_proc_address(name)));
}

} // namespace 

namespace scm {
namespace gl {
namespace opengl {

bool
gl_core::is_initialized() const
{
    return (_initialized);
}

bool
gl_core::is_supported(const std::string& ext) const
{
    if (_extensions.find(ext) != _extensions.end()) {
        return (true);
    }
    else {
        return (false);
    }
}

bool
gl_core::version_supported(unsigned in_major, unsigned in_minor) const
{
    unsigned avail_version =   _context_info._version_major * 1000
                             + _context_info._version_minor * 10;
    unsigned asked_version =   in_major * 1000
                             + in_minor * 10;

    return (avail_version >= asked_version);
}

const gl_core::context_info&
gl_core::context_information() const
{
    return (_context_info);
}

gl_core::gl_core()
{
    _initialized = false;

    version_1_0_available   = false;
    version_1_1_available   = false;
    version_1_2_available   = false;
    version_1_3_available   = false;
    version_1_4_available   = false;
    version_1_5_available   = false;
    version_2_0_available   = false;
    version_2_1_available   = false;
    version_3_0_available   = false;
    version_3_1_available   = false;
    version_3_2_available   = false;
    version_3_3_available   = false;
    version_4_0_available   = false;
    version_4_1_available   = false;
    version_4_2_available   = false;
    version_4_3_available   = false;
    version_4_4_available   = false;

    extension_ARB_bindless_texture              = false;
    extension_ARB_cl_event                      = false;
    extension_ARB_compute_variable_group_size   = false;
    extension_ARB_debug_output                  = false;
    extension_ARB_map_buffer_alignment          = false;
    extension_ARB_robustness                    = false;
    extension_ARB_shading_language_include      = false;
    extension_ARB_sparse_texture                = false;
    extension_ARB_texture_compression_bptc      = false;

    extension_EXT_direct_state_access_available = false;
    extension_EXT_shader_image_load_store       = false;
    extension_EXT_texture_compression_s3tc      = false;

    extension_NVX_gpu_memory_info               = false;
    extension_NV_bindless_texture               = false;
    extension_NV_shader_buffer_load             = false;

    extension_EXT_raster_multisample            = false;
    extension_NV_framebuffer_mixed_samples      = false;
    extension_NV_fragment_coverage_to_color     = false;
    extension_NV_sample_locations               = false;
    extension_NV_conservative_raster            = false;
    extension_EXT_post_depth_coverage           = false;
    extension_EXT_sparse_texture2               = false;
    extension_NV_shader_atomic_int64            = false;
    extension_NV_fragment_shader_interlock      = false;
    extension_NV_sample_mask_override_coverage  = false;
    extension_NV_fill_rectangle                 = false;
}

bool
gl_core::initialize()
{
    if (is_initialized()) {
        return (true);
    }
    log::logger_format_saver save_indent(glout().associated_logger());
    glout() << log::info << "gl_core::initialize(): starting to initialize gl core:" << log::end;
    glout() << log::indent;

    init_entry_points();

    if (!version_1_1_available) {
        glerr() << log::fatal << "gl_core::initialize(): unable to initialize gl core, missing vital entry points" << log::end;
        return (false);
    }

    { // parse the version string
        std::string gl_version_string(reinterpret_cast<const char*>(glGetString(GL_VERSION)));

        namespace qi = boost::spirit::qi;
        namespace ph = boost::phoenix;

        std::string::iterator b = gl_version_string.begin();
        std::string::iterator e = gl_version_string.end();

        qi::rule<std::string::iterator> gl_version_string_format =
                 qi::int_[ph::ref(_context_info._version_major) = qi::_1]
            >>   qi::char_('.')
            >>   qi::int_[ph::ref(_context_info._version_minor) = qi::_1]
            >> -(qi::char_('.') >> qi::int_[ph::ref(_context_info._version_release) = qi::_1])
            >> (*qi::char_)[ph::assign(ph::ref(_context_info._version_info), ph::begin(qi::_1), ph::end(qi::_1))];

        if (   !qi::phrase_parse(b, e, gl_version_string_format, boost::spirit::ascii::space)
            || b != e) {
            glerr() << log::error
                    << "gl_core::initialize(): "
                    << "unable to parse OpenGL Version string, malformed version string ('"
                    << gl_version_string << "')" << log::end;
            return (false);
        }
    }

    _context_info._vendor.assign(reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
    _context_info._renderer.assign(reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
    _context_info._glsl_version_info.assign(reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION)));

    if (_context_info._version_major == 3) {
        if (    _context_info._version_minor == 0 && !version_3_0_available) {
            glout() << log::warning << "gl_core::initialize(): OpenGL version 3.0 reported but missing entry points detected" << log::end;
        }
        if (    _context_info._version_minor == 1 && !version_3_1_available) {
            glout() << log::warning << "gl_core::initialize(): OpenGL version 3.1 reported but missing entry points detected" << log::end;
        }
        if (    _context_info._version_minor == 2 && !version_3_2_available) {
            glout() << log::warning << "gl_core::initialize(): OpenGL version 3.2 reported but missing entry points detected" << log::end;
        }
        if (    _context_info._version_minor == 3 && !version_3_3_available) {
            glout() << log::warning << "gl_core::initialize(): OpenGL version 3.3 reported but missing entry points detected" << log::end;
        }
    }
    else if (_context_info._version_major == 4) {
        if (    _context_info._version_minor == 0 && !version_4_0_available) {
            glout() << log::warning << "gl_core::initialize(): OpenGL version 4.0 reported but missing entry points detected" << log::end;
        }
        if (    _context_info._version_minor == 1 && !version_4_1_available) {
            glout() << log::warning << "gl_core::initialize(): OpenGL version 4.1 reported but missing entry points detected" << log::end;
        }
    }
    else if (_context_info._version_major < 3) {
        glerr() << log::fatal << "gl_core::initialize(): at least OpenGL version 3.0 requiered" << log::end;
        return (false);
    }

    // get the extension strings
    if (_context_info._version_major >= 3) {
        GLint num_extensions = 0;

        glGetIntegerv(GL_NUM_EXTENSIONS, &num_extensions);
        for (int i = 0; i < num_extensions; ++i) {
            const std::string extension_string(reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i)));
            _extensions.insert(extension_string);
        }

        int profile_mask;
        glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile_mask);
        if (profile_mask & GL_CONTEXT_CORE_PROFILE_BIT) {
            _context_info._profile = context_info::profile_core;
            _context_info._profile_string.assign("core profile");
        }
        else if (profile_mask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT) {
            _context_info._profile = context_info::profile_compatibility;
            _context_info._profile_string.assign("compatibility profile");
        }
        else {
            _context_info._profile_string.assign("unknown profile");
        }
    }
    //glout() << log::info
    //        << "OpenGL context information:" << log::nline
    //        << *this << log::end;

    //else {
    //    std::string gl_ext_string(reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS)));

    //    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    //    boost::char_separator<char> space_separator(" ");
    //    tokenizer                   extension_strings(gl_ext_string, space_separator);

    //    for (tokenizer::const_iterator i = extension_strings.begin(); i != extension_strings.end(); ++i) {
    //        _extensions.insert(*i);
    //    }
    //}
    if (is_supported("GL_ARB_cl_event") && !extension_ARB_cl_event) {
        glout() << log::warning << "gl_core::initialize(): GL_ARB_cl_event reported but missing entry points detected" << log::end;
    }
    if (is_supported("GL_ARB_debug_output") && !extension_ARB_debug_output) {
        glout() << log::warning << "gl_core::initialize(): GL_ARB_debug_output reported but missing entry points detected" << log::end;
    }
    if (is_supported("GL_ARB_robustness") && !extension_ARB_robustness) {
        glout() << log::warning << "gl_core::initialize(): ARB_robustness reported but missing entry points detected" << log::end;
    }

    extension_ARB_shading_language_include  = extension_ARB_shading_language_include  && is_supported("GL_ARB_shading_language_include");
    extension_ARB_cl_event                  = extension_ARB_cl_event                  && is_supported("GL_ARB_cl_event");
    extension_ARB_debug_output              = extension_ARB_debug_output              && is_supported("GL_ARB_debug_output");
    extension_ARB_robustness                = extension_ARB_robustness                && is_supported("GL_ARB_robustness");
    extension_EXT_shader_image_load_store   = extension_EXT_shader_image_load_store   && is_supported("GL_EXT_shader_image_load_store");

    extension_ARB_map_buffer_alignment      = is_supported("GL_ARB_map_buffer_alignment");
    extension_EXT_texture_compression_s3tc  = is_supported("GL_EXT_texture_compression_s3tc");
    extension_ARB_texture_compression_bptc  = is_supported("GL_ARB_texture_compression_bptc");
    extension_NVX_gpu_memory_info           = is_supported("GL_NVX_gpu_memory_info");

    extension_NV_bindless_texture           = extension_NV_bindless_texture           && is_supported("GL_NV_bindless_texture");
    extension_NV_shader_buffer_load         = extension_NV_shader_buffer_load         && is_supported("GL_NV_shader_buffer_load");

    extension_EXT_raster_multisample        = extension_EXT_raster_multisample        && is_supported("GL_EXT_raster_multisample");
    extension_NV_framebuffer_mixed_samples  = extension_NV_framebuffer_mixed_samples  && is_supported("GL_NV_framebuffer_mixed_samples");
    extension_NV_fragment_coverage_to_color = extension_NV_fragment_coverage_to_color && is_supported("GL_NV_fragment_coverage_to_color");
    extension_NV_sample_locations           = extension_NV_sample_locations           && is_supported("GL_NV_sample_locations");
    extension_NV_conservative_raster        = extension_NV_conservative_raster        && is_supported("GL_NV_conservative_raster");
    extension_EXT_post_depth_coverage       = is_supported("GL_EXT_post_depth_coverage");
    extension_EXT_sparse_texture2           = is_supported("GL_EXT_sparse_texture2");
    extension_NV_shader_atomic_int64        = is_supported("GL_NV_shader_atomic_int64");
    extension_NV_fragment_shader_interlock  = is_supported("GL_NV_fragment_shader_interlock");
    extension_NV_sample_mask_override_coverage = is_supported("GL_NV_sample_mask_override_coverage");
    extension_NV_fill_rectangle             = is_supported("GL_NV_fill_rectangle");

#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    if (!is_supported("GL_EXT_direct_state_access")) {
        glout() << log::warning
                << "gl_core::initialize(): "
                << "GL_EXT_direct_state_access not supported but enabled for scm_gl_core use "
                << "(undefine SCM_GL_CORE_USE_DIRECT_STATE_ACCESS!)" << log::end;
    }
    extension_EXT_direct_state_access_available = extension_EXT_direct_state_access_available && is_supported("GL_EXT_direct_state_access");

#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    glout() << log::outdent;
    glout() << log::info << "gl_core::initialize(): finished to initializing gl core." << log::end;

    return (true);
}

std::ostream& operator<<(std::ostream& out_stream, const gl_core& c)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_stream << "vendor:           " << c._context_info._vendor << std::endl
               << "renderer:         " << c._context_info._renderer << std::endl
               << "version:          " << c._context_info._version_major << "." 
                                       << c._context_info._version_minor << "." 
                                       << c._context_info._version_release;
    if (!c._context_info._version_info.empty())
         out_stream                    << " " << c._context_info._version_info;
    if (!c._context_info._profile_string.empty())
         out_stream                    << " " << c._context_info._profile_string;
    out_stream << std::endl;
    out_stream << "shading language: " << c._context_info._glsl_version_info;
    out_stream << std::endl;
    out_stream << "extensions :      " << "(found " << c._extensions.size() << ")" << std::endl;

    for (gl_core::string_set::const_iterator i = c._extensions.begin(); i != c._extensions.end(); ++i) {
        out_stream << "                  " << *i << std::endl;
    }

    return (out_stream);
}

void
gl_core::init_entry_points()
{
    /* Visual Studio regex find and replace
        find:           {PFN[^ ]+}[ ]+{gl[^;]+}
        replace with:   \2 = gl_proc_address<\1>("\2")
                        SCM_INIT_GL_ENTRY(\1, \2, "give context to error", init_success)
    */

#define SCM_INIT_GL_ENTRY(PFN, fun, ctx_str, errflag)                                                                            \
    if (0 == (fun = gl_proc_address<PFN>(#fun))) {                                                                               \
        errflag = false;                                                                                                         \
        glout() << log::warning << "- missing entry point (source: " << (ctx_str) << ", function: " << #fun << ")." << log::end; \
    }

    log::logger_format_saver save_indent(glout().associated_logger());
    glout() << log::info << "initializing function entry points..." << log::end;
    glout() << log::indent;


    bool init_success = true;

    // version 1.0 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLCULLFACEPROC, glCullFace, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRONTFACEPROC, glFrontFace, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLHINTPROC, glHint, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLLINEWIDTHPROC, glLineWidth, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTSIZEPROC, glPointSize, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOLYGONMODEPROC, glPolygonMode, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSCISSORPROC, glScissor, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERFPROC, glTexParameterf, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERFVPROC, glTexParameterfv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIPROC, glTexParameteri, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIVPROC, glTexParameteriv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE1DPROC, glTexImage1D, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE2DPROC, glTexImage2D, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWBUFFERPROC, glDrawBuffer, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARPROC, glClear, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARCOLORPROC, glClearColor, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARSTENCILPROC, glClearStencil, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARDEPTHPROC, glClearDepth, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILMASKPROC, glStencilMask, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOLORMASKPROC, glColorMask, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHMASKPROC, glDepthMask, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEPROC, glDisable, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENABLEPROC, glEnable, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFINISHPROC, glFinish, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFLUSHPROC, glFlush, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLENDFUNCPROC, glBlendFunc, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLLOGICOPPROC, glLogicOp, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILFUNCPROC, glStencilFunc, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILOPPROC, glStencilOp, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHFUNCPROC, glDepthFunc, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPIXELSTOREFPROC, glPixelStoref, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPIXELSTOREIPROC, glPixelStorei, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLREADBUFFERPROC, glReadBuffer, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLREADPIXELSPROC, glReadPixels, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBOOLEANVPROC, glGetBooleanv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETDOUBLEVPROC, glGetDoublev, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETERRORPROC, glGetError, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFLOATVPROC, glGetFloatv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGERVPROC, glGetIntegerv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSTRINGPROC, glGetString, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXIMAGEPROC, glGetTexImage, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERFVPROC, glGetTexParameterfv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERIVPROC, glGetTexParameteriv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXLEVELPARAMETERFVPROC, glGetTexLevelParameterfv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXLEVELPARAMETERIVPROC, glGetTexLevelParameteriv, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISENABLEDPROC, glIsEnabled, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHRANGEPROC, glDepthRange, "OpenGL Core 1.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVIEWPORTPROC, glViewport, "OpenGL Core 1.0", init_success);
    version_1_0_available = init_success;
                                            
    // version 1.1 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLDRAWARRAYSPROC, glDrawArrays, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSPROC, glDrawElements, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPOINTERVPROC, glGetPointerv, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOLYGONOFFSETPROC, glPolygonOffset, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXIMAGE1DPROC, glCopyTexImage1D, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXIMAGE2DPROC, glCopyTexImage2D, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXSUBIMAGE1DPROC, glCopyTexSubImage1D, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXSUBIMAGE2DPROC, glCopyTexSubImage2D, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSUBIMAGE1DPROC, glTexSubImage1D, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSUBIMAGE2DPROC, glTexSubImage2D, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDTEXTUREPROC, glBindTexture, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETETEXTURESPROC, glDeleteTextures, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENTEXTURESPROC, glGenTextures, "OpenGL Core 1.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISTEXTUREPROC, glIsTexture, "OpenGL Core 1.1", init_success);
    version_1_1_available = version_1_0_available && init_success;
                                            
    // version 1.2 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBLENDCOLORPROC, glBlendColor, "OpenGL Core 1.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLENDEQUATIONPROC, glBlendEquation, "OpenGL Core 1.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWRANGEELEMENTSPROC, glDrawRangeElements, "OpenGL Core 1.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE3DPROC, glTexImage3D, "OpenGL Core 1.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSUBIMAGE3DPROC, glTexSubImage3D, "OpenGL Core 1.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXSUBIMAGE3DPROC, glCopyTexSubImage3D, "OpenGL Core 1.2", init_success);
    version_1_2_available = version_1_1_available && init_success;
                                            
    // version 1.3 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLACTIVETEXTUREPROC, glActiveTexture, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLECOVERAGEPROC, glSampleCoverage, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXIMAGE3DPROC, glCompressedTexImage3D, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXIMAGE2DPROC, glCompressedTexImage2D, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXIMAGE1DPROC, glCompressedTexImage1D, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC, glCompressedTexSubImage3D, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC, glCompressedTexSubImage2D, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC, glCompressedTexSubImage1D, "OpenGL Core 1.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETCOMPRESSEDTEXIMAGEPROC, glGetCompressedTexImage, "OpenGL Core 1.3", init_success);
    version_1_3_available = version_1_2_available && init_success;
                                            
    // version 1.4 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBLENDFUNCSEPARATEPROC, glBlendFuncSeparate, "OpenGL Core 1.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWARRAYSPROC, glMultiDrawArrays, "OpenGL Core 1.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWELEMENTSPROC, glMultiDrawElements, "OpenGL Core 1.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERFPROC, glPointParameterf, "OpenGL Core 1.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERFVPROC, glPointParameterfv, "OpenGL Core 1.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERIPROC, glPointParameteri, "OpenGL Core 1.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERIVPROC, glPointParameteriv, "OpenGL Core 1.4", init_success);
    version_1_4_available = version_1_3_available && init_success;
                                            
    // version 1.5 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLGENQUERIESPROC, glGenQueries, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEQUERIESPROC, glDeleteQueries, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISQUERYPROC, glIsQuery, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBEGINQUERYPROC, glBeginQuery, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENDQUERYPROC, glEndQuery, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYIVPROC, glGetQueryiv, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTIVPROC, glGetQueryObjectiv, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTUIVPROC, glGetQueryObjectuiv, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERPROC, glBindBuffer, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEBUFFERSPROC, glDeleteBuffers, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENBUFFERSPROC, glGenBuffers, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISBUFFERPROC, glIsBuffer, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBUFFERDATAPROC, glBufferData, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBUFFERSUBDATAPROC, glBufferSubData, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERSUBDATAPROC, glGetBufferSubData, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAPBUFFERPROC, glMapBuffer, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNMAPBUFFERPROC, glUnmapBuffer, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERPARAMETERIVPROC, glGetBufferParameteriv, "OpenGL Core 1.5", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERPOINTERVPROC, glGetBufferPointerv, "OpenGL Core 1.5", init_success);
    version_1_5_available = version_1_4_available && init_success;

    // version 2.0 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBLENDEQUATIONSEPARATEPROC, glBlendEquationSeparate, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWBUFFERSPROC, glDrawBuffers, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILOPSEPARATEPROC, glStencilOpSeparate, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILFUNCSEPARATEPROC, glStencilFuncSeparate, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILMASKSEPARATEPROC, glStencilMaskSeparate, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLATTACHSHADERPROC, glAttachShader, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDATTRIBLOCATIONPROC, glBindAttribLocation, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPILESHADERPROC, glCompileShader, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCREATEPROGRAMPROC, glCreateProgram, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCREATESHADERPROC, glCreateShader, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEPROGRAMPROC, glDeleteProgram, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETESHADERPROC, glDeleteShader, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDETACHSHADERPROC, glDetachShader, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEVERTEXATTRIBARRAYPROC, glDisableVertexAttribArray, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEATTRIBPROC, glGetActiveAttrib, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMPROC, glGetActiveUniform, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETATTACHEDSHADERSPROC, glGetAttachedShaders, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETATTRIBLOCATIONPROC, glGetAttribLocation, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMIVPROC, glGetProgramiv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSHADERIVPROC, glGetShaderiv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSHADERSOURCEPROC, glGetShaderSource, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMFVPROC, glGetUniformfv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMIVPROC, glGetUniformiv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBDVPROC, glGetVertexAttribdv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBFVPROC, glGetVertexAttribfv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBIVPROC, glGetVertexAttribiv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBPOINTERVPROC, glGetVertexAttribPointerv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISPROGRAMPROC, glIsProgram, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISSHADERPROC, glIsShader, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLLINKPROGRAMPROC, glLinkProgram, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSHADERSOURCEPROC, glShaderSource, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUSEPROGRAMPROC, glUseProgram, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1FPROC, glUniform1f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2FPROC, glUniform2f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3FPROC, glUniform3f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4FPROC, glUniform4f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1IPROC, glUniform1i, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2IPROC, glUniform2i, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3IPROC, glUniform3i, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4IPROC, glUniform4i, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1FVPROC, glUniform1fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2FVPROC, glUniform2fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3FVPROC, glUniform3fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4FVPROC, glUniform4fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1IVPROC, glUniform1iv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2IVPROC, glUniform2iv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3IVPROC, glUniform3iv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4IVPROC, glUniform4iv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2FVPROC, glUniformMatrix2fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3FVPROC, glUniformMatrix3fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVALIDATEPROGRAMPROC, glValidateProgram, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1DPROC, glVertexAttrib1d, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1DVPROC, glVertexAttrib1dv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1FPROC, glVertexAttrib1f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1FVPROC, glVertexAttrib1fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1SPROC, glVertexAttrib1s, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1SVPROC, glVertexAttrib1sv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2DPROC, glVertexAttrib2d, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2DVPROC, glVertexAttrib2dv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2FPROC, glVertexAttrib2f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2FVPROC, glVertexAttrib2fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2SPROC, glVertexAttrib2s, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2SVPROC, glVertexAttrib2sv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3DPROC, glVertexAttrib3d, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3DVPROC, glVertexAttrib3dv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3FPROC, glVertexAttrib3f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3FVPROC, glVertexAttrib3fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3SPROC, glVertexAttrib3s, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3SVPROC, glVertexAttrib3sv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NBVPROC, glVertexAttrib4Nbv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NIVPROC, glVertexAttrib4Niv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NSVPROC, glVertexAttrib4Nsv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUBPROC, glVertexAttrib4Nub, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUBVPROC, glVertexAttrib4Nubv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUIVPROC, glVertexAttrib4Nuiv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUSVPROC, glVertexAttrib4Nusv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4BVPROC, glVertexAttrib4bv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4DPROC, glVertexAttrib4d, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4DVPROC, glVertexAttrib4dv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4FPROC, glVertexAttrib4f, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4FVPROC, glVertexAttrib4fv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4IVPROC, glVertexAttrib4iv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4SPROC, glVertexAttrib4s, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4SVPROC, glVertexAttrib4sv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4UBVPROC, glVertexAttrib4ubv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4UIVPROC, glVertexAttrib4uiv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4USVPROC, glVertexAttrib4usv, "OpenGL Core 2.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer, "OpenGL Core 2.0", init_success);
    version_2_0_available = version_1_5_available && init_success;
                             
    // version 2.1 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2X3FVPROC, glUniformMatrix2x3fv, "OpenGL Core 2.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3X2FVPROC, glUniformMatrix3x2fv, "OpenGL Core 2.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2X4FVPROC, glUniformMatrix2x4fv, "OpenGL Core 2.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4X2FVPROC, glUniformMatrix4x2fv, "OpenGL Core 2.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3X4FVPROC, glUniformMatrix3x4fv, "OpenGL Core 2.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4X3FVPROC, glUniformMatrix4x3fv, "OpenGL Core 2.1", init_success);
    version_2_1_available = version_2_0_available && init_success;

    // version 3.0 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLCOLORMASKIPROC, glColorMaski, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBOOLEANI_VPROC, glGetBooleani_v, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGERI_VPROC, glGetIntegeri_v, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENABLEIPROC, glEnablei, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEIPROC, glDisablei, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISENABLEDIPROC, glIsEnabledi, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBEGINTRANSFORMFEEDBACKPROC, glBeginTransformFeedback, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENDTRANSFORMFEEDBACKPROC, glEndTransformFeedback, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERRANGEPROC, glBindBufferRange, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERBASEPROC, glBindBufferBase, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTRANSFORMFEEDBACKVARYINGSPROC, glTransformFeedbackVaryings, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTRANSFORMFEEDBACKVARYINGPROC, glGetTransformFeedbackVarying, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLAMPCOLORPROC, glClampColor, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBEGINCONDITIONALRENDERPROC, glBeginConditionalRender, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENDCONDITIONALRENDERPROC, glEndConditionalRender, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBIPOINTERPROC, glVertexAttribIPointer, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBIIVPROC, glGetVertexAttribIiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBIUIVPROC, glGetVertexAttribIuiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1IPROC, glVertexAttribI1i, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2IPROC, glVertexAttribI2i, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3IPROC, glVertexAttribI3i, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4IPROC, glVertexAttribI4i, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1UIPROC, glVertexAttribI1ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2UIPROC, glVertexAttribI2ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3UIPROC, glVertexAttribI3ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4UIPROC, glVertexAttribI4ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1IVPROC, glVertexAttribI1iv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2IVPROC, glVertexAttribI2iv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3IVPROC, glVertexAttribI3iv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4IVPROC, glVertexAttribI4iv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1UIVPROC, glVertexAttribI1uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2UIVPROC, glVertexAttribI2uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3UIVPROC, glVertexAttribI3uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4UIVPROC, glVertexAttribI4uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4BVPROC, glVertexAttribI4bv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4SVPROC, glVertexAttribI4sv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4UBVPROC, glVertexAttribI4ubv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4USVPROC, glVertexAttribI4usv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMUIVPROC, glGetUniformuiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDFRAGDATALOCATIONPROC, glBindFragDataLocation, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAGDATALOCATIONPROC, glGetFragDataLocation, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1UIPROC, glUniform1ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2UIPROC, glUniform2ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3UIPROC, glUniform3ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4UIPROC, glUniform4ui, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1UIVPROC, glUniform1uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2UIVPROC, glUniform2uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3UIVPROC, glUniform3uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4UIVPROC, glUniform4uiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIIVPROC, glTexParameterIiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIUIVPROC, glTexParameterIuiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERIIVPROC, glGetTexParameterIiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERIUIVPROC, glGetTexParameterIuiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERIVPROC, glClearBufferiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERUIVPROC, glClearBufferuiv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERFVPROC, glClearBufferfv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERFIPROC, glClearBufferfi, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSTRINGIPROC, glGetStringi, "OpenGL Core 3.0", init_success);
    // use ARB_framebuffer_object
    SCM_INIT_GL_ENTRY(PFNGLISRENDERBUFFERPROC, glIsRenderbuffer, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDRENDERBUFFERPROC, glBindRenderbuffer, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETERENDERBUFFERSPROC, glDeleteRenderbuffers, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENRENDERBUFFERSPROC, glGenRenderbuffers, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLRENDERBUFFERSTORAGEPROC, glRenderbufferStorage, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETRENDERBUFFERPARAMETERIVPROC, glGetRenderbufferParameteriv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISFRAMEBUFFERPROC, glIsFramebuffer, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURE1DPROC, glFramebufferTexture1D, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURE3DPROC, glFramebufferTexture3D, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERRENDERBUFFERPROC, glFramebufferRenderbuffer, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC, glGetFramebufferAttachmentParameteriv, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENERATEMIPMAPPROC, glGenerateMipmap, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLITFRAMEBUFFERPROC, glBlitFramebuffer, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC, glRenderbufferStorageMultisample, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURELAYERPROC, glFramebufferTextureLayer, "OpenGL Core 3.0", init_success);
    // use ARB_map_buffer_ranger
    SCM_INIT_GL_ENTRY(PFNGLMAPBUFFERRANGEPROC, glMapBufferRange, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFLUSHMAPPEDBUFFERRANGEPROC, glFlushMappedBufferRange, "OpenGL Core 3.0", init_success);
    // use ARB_vertex_array_object          
    SCM_INIT_GL_ENTRY(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays, "OpenGL Core 3.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISVERTEXARRAYPROC, glIsVertexArray, "OpenGL Core 3.0", init_success);
    version_3_0_available = version_2_1_available && init_success;

    // version 3.1 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLDRAWARRAYSINSTANCEDPROC, glDrawArraysInstanced, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSINSTANCEDPROC, glDrawElementsInstanced, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXBUFFERPROC, glTexBuffer, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPRIMITIVERESTARTINDEXPROC, glPrimitiveRestartIndex, "OpenGL Core 3.1", init_success);
    // use ARB_copy_buffer                  
    SCM_INIT_GL_ENTRY(PFNGLCOPYBUFFERSUBDATAPROC, glCopyBufferSubData, "OpenGL Core 3.1", init_success);
    // use ARB_uniform_buffer_object        
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMINDICESPROC, glGetUniformIndices, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMSIVPROC, glGetActiveUniformsiv, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMNAMEPROC, glGetActiveUniformName, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMBLOCKINDEXPROC, glGetUniformBlockIndex, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMBLOCKIVPROC, glGetActiveUniformBlockiv, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC, glGetActiveUniformBlockName, "OpenGL Core 3.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMBLOCKBINDINGPROC, glUniformBlockBinding, "OpenGL Core 3.1", init_success);
    version_3_1_available = version_3_0_available && init_success;

    // version 3.2 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGER64I_VPROC, glGetInteger64i_v, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERPARAMETERI64VPROC, glGetBufferParameteri64v, "OpenGL Core 3.2", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMPARAMETERIPROC, glProgramParameteri, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTUREPROC, glFramebufferTexture, "OpenGL Core 3.2", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTUREFACEPROC, glFramebufferTextureFace, "OpenGL Core 3.2", init_success);
    // use ARB_draw_elements_base_vertex    
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSBASEVERTEXPROC, glDrawElementsBaseVertex, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC, glDrawRangeElementsBaseVertex, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC, glDrawElementsInstancedBaseVertex, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC, glMultiDrawElementsBaseVertex, "OpenGL Core 3.2", init_success);
    // use ARB_provoking_vertex             
    SCM_INIT_GL_ENTRY(PFNGLPROVOKINGVERTEXPROC, glProvokingVertex, "OpenGL Core 3.2", init_success);
    // use ARB_sync                         
    SCM_INIT_GL_ENTRY(PFNGLFENCESYNCPROC, glFenceSync, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISSYNCPROC, glIsSync, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETESYNCPROC, glDeleteSync, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLIENTWAITSYNCPROC, glClientWaitSync, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLWAITSYNCPROC, glWaitSync, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGER64VPROC, glGetInteger64v, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSYNCIVPROC, glGetSynciv, "OpenGL Core 3.2", init_success);
    // use ARB_texture_multisample          
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE2DMULTISAMPLEPROC, glTexImage2DMultisample, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE3DMULTISAMPLEPROC, glTexImage3DMultisample, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETMULTISAMPLEFVPROC, glGetMultisamplefv, "OpenGL Core 3.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLEMASKIPROC, glSampleMaski, "OpenGL Core 3.2", init_success);
    version_3_2_available = version_3_1_available && init_success;

    // version 3.3 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    // use GL_ARB_shader_bit_encoding
    // use ARB_blend_func_extended
    SCM_INIT_GL_ENTRY(PFNGLBINDFRAGDATALOCATIONINDEXEDPROC, glBindFragDataLocationIndexed, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAGDATAINDEXPROC, glGetFragDataIndex, "OpenGL Core 3.3", init_success);
    // use GL_ARB_explicit_attrib_location
    // use GL_ARB_occlusion_query2
    // use ARB_sampler_objects
    SCM_INIT_GL_ENTRY(PFNGLGENSAMPLERSPROC, glGenSamplers, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETESAMPLERSPROC, glDeleteSamplers, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISSAMPLERPROC, glIsSampler, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDSAMPLERPROC, glBindSampler, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIPROC, glSamplerParameteri, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIVPROC, glSamplerParameteriv, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERFPROC, glSamplerParameterf, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERFVPROC, glSamplerParameterfv, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIIVPROC, glSamplerParameterIiv, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIUIVPROC, glSamplerParameterIuiv, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERIVPROC, glGetSamplerParameteriv, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERIIVPROC, glGetSamplerParameterIiv, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERFVPROC, glGetSamplerParameterfv, "OpenGL Core 3.3", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERIFVPROC, glGetSamplerParameterIfv, "OpenGL Core 3.3", init_success);
    // use GL_ARB_texture_rgb10_a2ui
    // use GL_ARB_texture_swizzle
    // use ARB_timer_query
    SCM_INIT_GL_ENTRY(PFNGLQUERYCOUNTERPROC, glQueryCounter, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTI64VPROC, glGetQueryObjecti64v, "OpenGL Core 3.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTUI64VPROC, glGetQueryObjectui64v, "OpenGL Core 3.3", init_success);
    // use GL_ARB_texture_swizzle
    /// TODO missing entry points
    // use ARB_vertex_type_2_10_10_10_rev
    // non which concern core profile
    version_3_3_available = version_3_2_available && init_success;

    // version 4.0 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    // use ARB_draw_buffers_blend
    SCM_INIT_GL_ENTRY(PFNGLBLENDEQUATIONIPROC, glBlendEquationi, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLENDEQUATIONSEPARATEIPROC, glBlendEquationSeparatei, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLENDFUNCIPROC, glBlendFunci, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLENDFUNCSEPARATEIPROC, glBlendFuncSeparatei, "OpenGL Core 4.0", init_success);
    // use ARB_draw_indirect
    SCM_INIT_GL_ENTRY(PFNGLDRAWARRAYSINDIRECTPROC, glDrawArraysIndirect, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSINDIRECTPROC, glDrawElementsIndirect, "OpenGL Core 4.0", init_success);
    // use ARB_gpu_shader5
    // use ARB_gpu_shader_fp64
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1DPROC, glUniform1d, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2DPROC, glUniform2d, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3DPROC, glUniform3d, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4DPROC, glUniform4d, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1DVPROC, glUniform1dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2DVPROC, glUniform2dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3DVPROC, glUniform3dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4DVPROC, glUniform4dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2DVPROC, glUniformMatrix2dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3DVPROC, glUniformMatrix3dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4DVPROC, glUniformMatrix4dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2X3DVPROC, glUniformMatrix2x3dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2X4DVPROC, glUniformMatrix2x4dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3X2DVPROC, glUniformMatrix3x2dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3X4DVPROC, glUniformMatrix3x4dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4X2DVPROC, glUniformMatrix4x2dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4X3DVPROC, glUniformMatrix4x3dv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMDVPROC, glGetUniformdv, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1DEXTPROC, glProgramUniform1dEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2DEXTPROC, glProgramUniform2dEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3DEXTPROC, glProgramUniform3dEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4DEXTPROC, glProgramUniform4dEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1DVEXTPROC, glProgramUniform1dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2DVEXTPROC, glProgramUniform2dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3DVEXTPROC, glProgramUniform3dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4DVEXTPROC, glProgramUniform4dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2DVEXTPROC, glProgramUniformMatrix2dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3DVEXTPROC, glProgramUniformMatrix3dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4DVEXTPROC, glProgramUniformMatrix4dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X3DVEXTPROC, glProgramUniformMatrix2x3dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X4DVEXTPROC, glProgramUniformMatrix2x4dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X2DVEXTPROC, glProgramUniformMatrix3x2dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X4DVEXTPROC, glProgramUniformMatrix3x4dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X2DVEXTPROC, glProgramUniformMatrix4x2dvEXT, "OpenGL Core 4.0", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X3DVEXTPROC, glProgramUniformMatrix4x3dvEXT, "OpenGL Core 4.0", init_success);
    // use ARB_sample_shading
    SCM_INIT_GL_ENTRY(PFNGLMINSAMPLESHADINGPROC, glMinSampleShading, "OpenGL Core 4.0", init_success);
    // use ARB_shader_subroutine
    SCM_INIT_GL_ENTRY(PFNGLGETSUBROUTINEUNIFORMLOCATIONPROC, glGetSubroutineUniformLocation, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSUBROUTINEINDEXPROC, glGetSubroutineIndex, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVESUBROUTINEUNIFORMIVPROC, glGetActiveSubroutineUniformiv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVESUBROUTINEUNIFORMNAMEPROC, glGetActiveSubroutineUniformName, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVESUBROUTINENAMEPROC, glGetActiveSubroutineName, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMSUBROUTINESUIVPROC, glUniformSubroutinesuiv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMSUBROUTINEUIVPROC, glGetUniformSubroutineuiv, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMSTAGEIVPROC, glGetProgramStageiv, "OpenGL Core 4.0", init_success);
    // use ARB_tessellation_shader
    SCM_INIT_GL_ENTRY(PFNGLPATCHPARAMETERIPROC, glPatchParameteri, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPATCHPARAMETERFVPROC, glPatchParameterfv, "OpenGL Core 4.0", init_success);
    // use ARB_texture_buffer_object_rgb32
    // use ARB_texture_cube_map_array
    // use ARB_transform_feedback2
    SCM_INIT_GL_ENTRY(PFNGLBINDTRANSFORMFEEDBACKPROC, glBindTransformFeedback, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETETRANSFORMFEEDBACKSPROC, glDeleteTransformFeedbacks, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENTRANSFORMFEEDBACKSPROC, glGenTransformFeedbacks, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISTRANSFORMFEEDBACKPROC, glIsTransformFeedback, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPAUSETRANSFORMFEEDBACKPROC, glPauseTransformFeedback, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLRESUMETRANSFORMFEEDBACKPROC, glResumeTransformFeedback, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWTRANSFORMFEEDBACKPROC, glDrawTransformFeedback, "OpenGL Core 4.0", init_success);
    // use ARB_transform_feedback3
    SCM_INIT_GL_ENTRY(PFNGLDRAWTRANSFORMFEEDBACKSTREAMPROC, glDrawTransformFeedbackStream, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBEGINQUERYINDEXEDPROC, glBeginQueryIndexed, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENDQUERYINDEXEDPROC, glEndQueryIndexed, "OpenGL Core 4.0", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYINDEXEDIVPROC, glGetQueryIndexediv, "OpenGL Core 4.0", init_success);
    version_4_0_available = version_3_3_available && init_success;

    // version 4.1 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    // use  ARB_ES2_compatibility
    SCM_INIT_GL_ENTRY(PFNGLRELEASESHADERCOMPILERPROC, glReleaseShaderCompiler, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSHADERBINARYPROC, glShaderBinary, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSHADERPRECISIONFORMATPROC, glGetShaderPrecisionFormat, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHRANGEFPROC, glDepthRangef, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARDEPTHFPROC, glClearDepthf, "OpenGL Core 4.1", init_success);
    // use ARB_get_program_binary
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMBINARYPROC, glGetProgramBinary, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMBINARYPROC, glProgramBinary, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMPARAMETERIPROC, glProgramParameteri, "OpenGL Core 4.1", init_success);
    // use ARB_separate_shader_objects
    SCM_INIT_GL_ENTRY(PFNGLUSEPROGRAMSTAGESPROC, glUseProgramStages, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLACTIVESHADERPROGRAMPROC, glActiveShaderProgram, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCREATESHADERPROGRAMVPROC, glCreateShaderProgramv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDPROGRAMPIPELINEPROC, glBindProgramPipeline, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEPROGRAMPIPELINESPROC, glDeleteProgramPipelines, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENPROGRAMPIPELINESPROC, glGenProgramPipelines, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISPROGRAMPIPELINEPROC, glIsProgramPipeline, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMPIPELINEIVPROC, glGetProgramPipelineiv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1IPROC, glProgramUniform1i, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1IVPROC, glProgramUniform1iv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1FPROC, glProgramUniform1f, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1FVPROC, glProgramUniform1fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1DPROC, glProgramUniform1d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1DVPROC, glProgramUniform1dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1UIPROC, glProgramUniform1ui, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1UIVPROC, glProgramUniform1uiv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2IPROC, glProgramUniform2i, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2IVPROC, glProgramUniform2iv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2FPROC, glProgramUniform2f, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2FVPROC, glProgramUniform2fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2DPROC, glProgramUniform2d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2DVPROC, glProgramUniform2dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2UIPROC, glProgramUniform2ui, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2UIVPROC, glProgramUniform2uiv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3IPROC, glProgramUniform3i, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3IVPROC, glProgramUniform3iv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3FPROC, glProgramUniform3f, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3FVPROC, glProgramUniform3fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3DPROC, glProgramUniform3d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3DVPROC, glProgramUniform3dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3UIPROC, glProgramUniform3ui, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3UIVPROC, glProgramUniform3uiv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4IPROC, glProgramUniform4i, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4IVPROC, glProgramUniform4iv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4FPROC, glProgramUniform4f, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4FVPROC, glProgramUniform4fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4DPROC, glProgramUniform4d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4DVPROC, glProgramUniform4dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4UIPROC, glProgramUniform4ui, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4UIVPROC, glProgramUniform4uiv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2FVPROC, glProgramUniformMatrix2fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3FVPROC, glProgramUniformMatrix3fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4FVPROC, glProgramUniformMatrix4fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2DVPROC, glProgramUniformMatrix2dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3DVPROC, glProgramUniformMatrix3dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4DVPROC, glProgramUniformMatrix4dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC, glProgramUniformMatrix2x3fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC, glProgramUniformMatrix3x2fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC, glProgramUniformMatrix2x4fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC, glProgramUniformMatrix4x2fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC, glProgramUniformMatrix3x4fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC, glProgramUniformMatrix4x3fv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X3DVPROC, glProgramUniformMatrix2x3dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X2DVPROC, glProgramUniformMatrix3x2dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X4DVPROC, glProgramUniformMatrix2x4dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X2DVPROC, glProgramUniformMatrix4x2dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X4DVPROC, glProgramUniformMatrix3x4dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X3DVPROC, glProgramUniformMatrix4x3dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVALIDATEPROGRAMPIPELINEPROC, glValidateProgramPipeline, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMPIPELINEINFOLOGPROC, glGetProgramPipelineInfoLog, "OpenGL Core 4.1", init_success);
    // use ARB_shader_precision (no entry points)
    // use ARB_vertex_attrib_64bit
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL1DPROC, glVertexAttribL1d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL2DPROC, glVertexAttribL2d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL3DPROC, glVertexAttribL3d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL4DPROC, glVertexAttribL4d, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL1DVPROC, glVertexAttribL1dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL2DVPROC, glVertexAttribL2dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL3DVPROC, glVertexAttribL3dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL4DVPROC, glVertexAttribL4dv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBLPOINTERPROC, glVertexAttribLPointer, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBLDVPROC, glGetVertexAttribLdv, "OpenGL Core 4.1", init_success);
    // use ARB_viewport_array
    SCM_INIT_GL_ENTRY(PFNGLVIEWPORTARRAYVPROC, glViewportArrayv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVIEWPORTINDEXEDFPROC, glViewportIndexedf, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVIEWPORTINDEXEDFVPROC, glViewportIndexedfv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSCISSORARRAYVPROC, glScissorArrayv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSCISSORINDEXEDPROC, glScissorIndexed, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLSCISSORINDEXEDVPROC, glScissorIndexedv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHRANGEARRAYVPROC, glDepthRangeArrayv, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHRANGEINDEXEDPROC, glDepthRangeIndexed, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFLOATI_VPROC, glGetFloati_v, "OpenGL Core 4.1", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETDOUBLEI_VPROC, glGetDoublei_v, "OpenGL Core 4.1", init_success);
    version_4_1_available = version_4_0_available && init_success;

    // version 4.2 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    // ARB_base_instance
    SCM_INIT_GL_ENTRY(PFNGLDRAWARRAYSINSTANCEDBASEINSTANCEPROC, glDrawArraysInstancedBaseInstance, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSINSTANCEDBASEINSTANCEPROC, glDrawElementsInstancedBaseInstance, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXBASEINSTANCEPROC, glDrawElementsInstancedBaseVertexBaseInstance, "OpenGL Core 4.2", init_success);
    // ARB_shading_language_420pack (no entry points)
    // ARB_transform_feedback_instanced
    SCM_INIT_GL_ENTRY(PFNGLDRAWTRANSFORMFEEDBACKINSTANCEDPROC, glDrawTransformFeedbackInstanced, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWTRANSFORMFEEDBACKSTREAMINSTANCEDPROC, glDrawTransformFeedbackStreamInstanced, "OpenGL Core 4.2", init_success);
    // ARB_compressed_texture_pixel_storage (no entry points)
    // ARB_conservative_depth (no entry points)
    // ARB_internalformat_query
    SCM_INIT_GL_ENTRY(PFNGLGETINTERNALFORMATIVPROC, glGetInternalformativ, "OpenGL Core 4.2", init_success);
    // ARB_map_buffer_alignment (no entry points)
    // ARB_shader_atomic_counters
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEATOMICCOUNTERBUFFERIVPROC, glGetActiveAtomicCounterBufferiv, "OpenGL Core 4.2", init_success);
    // ARB_shader_image_load_store
    SCM_INIT_GL_ENTRY(PFNGLBINDIMAGETEXTUREPROC, glBindImageTexture, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMEMORYBARRIERPROC, glMemoryBarrier, "OpenGL Core 4.2", init_success);
    // ARB_shading_language_packing (no entry points)
    // ARB_texture_storage
    SCM_INIT_GL_ENTRY(PFNGLTEXSTORAGE1DPROC, glTexStorage1D, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSTORAGE2DPROC, glTexStorage2D, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSTORAGE3DPROC, glTexStorage3D, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESTORAGE1DEXTPROC, glTextureStorage1DEXT, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESTORAGE2DEXTPROC, glTextureStorage2DEXT, "OpenGL Core 4.2", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESTORAGE3DEXTPROC, glTextureStorage3DEXT, "OpenGL Core 4.2", init_success);
    version_4_2_available = version_4_1_available && init_success;

    // version 4.3 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    // ARB_arrays_of_arrays (no entry points, GLSL only)
    // ARB_fragment_layer_viewport (no entry points, GLSL only)
    // ARB_shader_image_size (no entry points, GLSL only)
    // ARB_ES3_compatibility (no entry points)
    // ARB_clear_buffer_object
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERDATAPROC, glClearBufferData, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERSUBDATAPROC, glClearBufferSubData, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARNAMEDBUFFERDATAEXTPROC, glClearNamedBufferDataEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARNAMEDBUFFERSUBDATAEXTPROC, glClearNamedBufferSubDataEXT, "OpenGL Core 4.3", init_success);
    // ARB_compute_shader
    SCM_INIT_GL_ENTRY(PFNGLDISPATCHCOMPUTEPROC, glDispatchCompute, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISPATCHCOMPUTEINDIRECTPROC, glDispatchComputeIndirect, "OpenGL Core 4.3", init_success);
    // ARB_copy_image
    SCM_INIT_GL_ENTRY(PFNGLCOPYIMAGESUBDATAPROC, glCopyImageSubData, "OpenGL Core 4.3", init_success);
    // KHR_debug (includes ARB_debug_output commands promoted to KHR without suffixes)
    SCM_INIT_GL_ENTRY(PFNGLDEBUGMESSAGECONTROLPROC, glDebugMessageControl, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEBUGMESSAGEINSERTPROC, glDebugMessageInsert, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEBUGMESSAGECALLBACKPROC, glDebugMessageCallback, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETDEBUGMESSAGELOGPROC, glGetDebugMessageLog, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPUSHDEBUGGROUPPROC, glPushDebugGroup, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOPDEBUGGROUPPROC, glPopDebugGroup, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLOBJECTLABELPROC, glObjectLabel, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETOBJECTLABELPROC, glGetObjectLabel, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLOBJECTPTRLABELPROC, glObjectPtrLabel, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETOBJECTPTRLABELPROC, glGetObjectPtrLabel, "OpenGL Core 4.3", init_success);
    // ARB_explicit_uniform_location (no entry points)
    // ARB_framebuffer_no_attachments
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERPARAMETERIPROC, glFramebufferParameteri, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAMEBUFFERPARAMETERIVPROC, glGetFramebufferParameteriv, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERPARAMETERIEXTPROC, glNamedFramebufferParameteriEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDFRAMEBUFFERPARAMETERIVEXTPROC, glGetNamedFramebufferParameterivEXT, "OpenGL Core 4.3", init_success);
    // ARB_internalformat_query2
    SCM_INIT_GL_ENTRY(PFNGLGETINTERNALFORMATI64VPROC, glGetInternalformati64v, "OpenGL Core 4.3", init_success);
    // ARB_invalidate_subdata
    SCM_INIT_GL_ENTRY(PFNGLINVALIDATETEXSUBIMAGEPROC, glInvalidateTexSubImage, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLINVALIDATETEXIMAGEPROC, glInvalidateTexImage, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLINVALIDATEBUFFERSUBDATAPROC, glInvalidateBufferSubData, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLINVALIDATEBUFFERDATAPROC, glInvalidateBufferData, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLINVALIDATEFRAMEBUFFERPROC, glInvalidateFramebuffer, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLINVALIDATESUBFRAMEBUFFERPROC, glInvalidateSubFramebuffer, "OpenGL Core 4.3", init_success);
    // ARB_multi_draw_indirect
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWARRAYSINDIRECTPROC, glMultiDrawArraysIndirect, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWELEMENTSINDIRECTPROC, glMultiDrawElementsIndirect, "OpenGL Core 4.3", init_success);
    // ARB_program_interface_query
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMINTERFACEIVPROC, glGetProgramInterfaceiv, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMRESOURCEINDEXPROC, glGetProgramResourceIndex, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMRESOURCENAMEPROC, glGetProgramResourceName, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMRESOURCEIVPROC, glGetProgramResourceiv, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMRESOURCELOCATIONPROC, glGetProgramResourceLocation, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMRESOURCELOCATIONINDEXPROC, glGetProgramResourceLocationIndex, "OpenGL Core 4.3", init_success);
    // ARB_robust_buffer_access_behavior (no entry points)
    // ARB_shader_storage_buffer_object
    SCM_INIT_GL_ENTRY(PFNGLSHADERSTORAGEBLOCKBINDINGPROC, glShaderStorageBlockBinding, "OpenGL Core 4.3", init_success);
    // ARB_stencil_texturing (no entry points)
    // ARB_texture_buffer_range
    SCM_INIT_GL_ENTRY(PFNGLTEXBUFFERRANGEPROC, glTexBufferRange, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREBUFFERRANGEEXTPROC, glTextureBufferRangeEXT, "OpenGL Core 4.3", init_success);
    // ARB_texture_query_levels (no entry points)
    // ARB_texture_storage_multisample
    SCM_INIT_GL_ENTRY(PFNGLTEXSTORAGE2DMULTISAMPLEPROC, glTexStorage2DMultisample, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSTORAGE3DMULTISAMPLEPROC, glTexStorage3DMultisample, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESTORAGE2DMULTISAMPLEEXTPROC, glTextureStorage2DMultisampleEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESTORAGE3DMULTISAMPLEEXTPROC, glTextureStorage3DMultisampleEXT, "OpenGL Core 4.3", init_success);
    // ARB_texture_view
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREVIEWPROC, glTextureView, "OpenGL Core 4.3", init_success);
    // ARB_vertex_attrib_binding
    SCM_INIT_GL_ENTRY(PFNGLBINDVERTEXBUFFERPROC, glBindVertexBuffer, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBFORMATPROC, glVertexAttribFormat, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBIFORMATPROC, glVertexAttribIFormat, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBLFORMATPROC, glVertexAttribLFormat, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBBINDINGPROC, glVertexAttribBinding, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXBINDINGDIVISORPROC, glVertexBindingDivisor, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYBINDVERTEXBUFFEREXTPROC, glVertexArrayBindVertexBufferEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYVERTEXATTRIBFORMATEXTPROC, glVertexArrayVertexAttribFormatEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYVERTEXATTRIBIFORMATEXTPROC, glVertexArrayVertexAttribIFormatEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYVERTEXATTRIBLFORMATEXTPROC, glVertexArrayVertexAttribLFormatEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYVERTEXATTRIBBINDINGEXTPROC, glVertexArrayVertexAttribBindingEXT, "OpenGL Core 4.3", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYVERTEXBINDINGDIVISOREXTPROC, glVertexArrayVertexBindingDivisorEXT, "OpenGL Core 4.3", init_success);
    version_4_3_available = version_4_2_available && init_success;

    // version 4.4 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBUFFERSTORAGEPROC, glBufferStorage, "OpenGL Core 4.4", init_success);
    // ARB_clear_texture
    SCM_INIT_GL_ENTRY(PFNGLCLEARTEXIMAGEPROC, glClearTexImage, "OpenGL Core 4.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARTEXSUBIMAGEPROC, glClearTexSubImage, "OpenGL Core 4.4", init_success);
    // ARB_enhanced_layouts (no entry points)
    // ARB_multi_bind
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERSBASEPROC, glBindBuffersBase, "OpenGL Core 4.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERSRANGEPROC, glBindBuffersRange, "OpenGL Core 4.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDTEXTURESPROC, glBindTextures, "OpenGL Core 4.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDSAMPLERSPROC, glBindSamplers, "OpenGL Core 4.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDIMAGETEXTURESPROC, glBindImageTextures, "OpenGL Core 4.4", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDVERTEXBUFFERSPROC, glBindVertexBuffers, "OpenGL Core 4.4", init_success);
    version_4_4_available = version_4_3_available && init_success;

    // GL_ARB_shading_language_include
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLNAMEDSTRINGARBPROC, glNamedStringARB, "GL_ARB_shading_language_include", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETENAMEDSTRINGARBPROC, glDeleteNamedStringARB, "GL_ARB_shading_language_include", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPILESHADERINCLUDEARBPROC, glCompileShaderIncludeARB, "GL_ARB_shading_language_include", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISNAMEDSTRINGARBPROC, glIsNamedStringARB, "GL_ARB_shading_language_include", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDSTRINGARBPROC, glGetNamedStringARB, "GL_ARB_shading_language_include", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDSTRINGIVARBPROC, glGetNamedStringivARB, "GL_ARB_shading_language_include", init_success);
    extension_ARB_shading_language_include = init_success;

    // ARB_cl_event ///////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLCREATESYNCFROMCLEVENTARBPROC, glCreateSyncFromCLeventARB, "ARB_cl_event", init_success);
    extension_ARB_cl_event = init_success;

    // ARB_debug_output ///////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLDEBUGMESSAGECONTROLARBPROC, glDebugMessageControlARB, "ARB_debug_output", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEBUGMESSAGEINSERTARBPROC, glDebugMessageInsertARB, "ARB_debug_output", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEBUGMESSAGECALLBACKARBPROC, glDebugMessageCallbackARB, "ARB_debug_output", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETDEBUGMESSAGELOGARBPROC, glGetDebugMessageLogARB, "ARB_debug_output", init_success);
    extension_ARB_debug_output = init_success;

    // ARB_robustness /////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLGETGRAPHICSRESETSTATUSARBPROC, glGetGraphicsResetStatusARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNMAPDVARBPROC, glGetnMapdvARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNMAPFVARBPROC, glGetnMapfvARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNMAPIVARBPROC, glGetnMapivARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNPIXELMAPFVARBPROC, glGetnPixelMapfvARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNPIXELMAPUIVARBPROC, glGetnPixelMapuivARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNPIXELMAPUSVARBPROC, glGetnPixelMapusvARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNPOLYGONSTIPPLEARBPROC, glGetnPolygonStippleARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNCOLORTABLEARBPROC, glGetnColorTableARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNCONVOLUTIONFILTERARBPROC, glGetnConvolutionFilterARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNSEPARABLEFILTERARBPROC, glGetnSeparableFilterARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNHISTOGRAMARBPROC, glGetnHistogramARB, "ARB_robustness", init_success);
    //SCM_INIT_GL_ENTRY(PFNGLGETNMINMAXARBPROC, glGetnMinmaxARB, "ARB_robustness", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNTEXIMAGEARBPROC, glGetnTexImageARB, "ARB_robustness", init_success);
    SCM_INIT_GL_ENTRY(PFNGLREADNPIXELSARBPROC, glReadnPixelsARB, "ARB_robustness", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNCOMPRESSEDTEXIMAGEARBPROC, glGetnCompressedTexImageARB, "ARB_robustness", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNUNIFORMFVARBPROC, glGetnUniformfvARB, "ARB_robustness", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNUNIFORMIVARBPROC, glGetnUniformivARB, "ARB_robustness", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNUNIFORMUIVARBPROC, glGetnUniformuivARB, "ARB_robustness", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNUNIFORMDVARBPROC, glGetnUniformdvARB, "ARB_robustness", init_success);
    extension_ARB_robustness = init_success;

    // EXT_shader_image_load_store ////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBINDIMAGETEXTUREEXTPROC, glBindImageTextureEXT, "EXT_shader_image_load_store", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMEMORYBARRIEREXTPROC, glMemoryBarrierEXT, "EXT_shader_image_load_store", init_success);
    extension_EXT_shader_image_load_store = init_success;

    // EXT_direct_state_access ////////////////////////////////////////////////////////////////////
    init_success = true;
    // use GL_EXT_draw_buffers2
    SCM_INIT_GL_ENTRY(PFNGLENABLEINDEXEDEXTPROC, glEnableIndexedEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEINDEXEDEXTPROC, glDisableIndexedEXT, "EXT_direct_state_access", init_success);

    // buffer handling ////////////////////////////////////////////////////////////////////////////
    SCM_INIT_GL_ENTRY(PFNGLNAMEDBUFFERDATAEXTPROC, glNamedBufferDataEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDBUFFERSUBDATAEXTPROC, glNamedBufferSubDataEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAPNAMEDBUFFEREXTPROC, glMapNamedBufferEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNMAPNAMEDBUFFEREXTPROC, glUnmapNamedBufferEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC, glGetNamedBufferParameterivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDBUFFERPOINTERVEXTPROC, glGetNamedBufferPointervEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDBUFFERSUBDATAEXTPROC, glGetNamedBufferSubDataEXT, "EXT_direct_state_access", init_success);

    SCM_INIT_GL_ENTRY(PFNGLMAPNAMEDBUFFERRANGEEXTPROC, glMapNamedBufferRangeEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC, glFlushMappedNamedBufferRangeEXT, "EXT_direct_state_access", init_success);

    SCM_INIT_GL_ENTRY(PFNGLNAMEDCOPYBUFFERSUBDATAEXTPROC, glNamedCopyBufferSubDataEXT, "EXT_direct_state_access", init_success);

    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYVERTEXATTRIBOFFSETEXTPROC, glVertexArrayVertexAttribOffsetEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXARRAYVERTEXATTRIBIOFFSETEXTPROC, glVertexArrayVertexAttribIOffsetEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLENABLEVERTEXARRAYATTRIBEXTPROC, glEnableVertexArrayAttribEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEVERTEXARRAYATTRIBEXTPROC, glDisableVertexArrayAttribEXT, "EXT_direct_state_access", init_success);

    // shader handling ////////////////////////////////////////////////////////////////////////////
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1FEXTPROC, glProgramUniform1fEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2FEXTPROC, glProgramUniform2fEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3FEXTPROC, glProgramUniform3fEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4FEXTPROC, glProgramUniform4fEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1IEXTPROC, glProgramUniform1iEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2IEXTPROC, glProgramUniform2iEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3IEXTPROC, glProgramUniform3iEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4IEXTPROC, glProgramUniform4iEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1FVEXTPROC, glProgramUniform1fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2FVEXTPROC, glProgramUniform2fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3FVEXTPROC, glProgramUniform3fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4FVEXTPROC, glProgramUniform4fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1IVEXTPROC, glProgramUniform1ivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2IVEXTPROC, glProgramUniform2ivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3IVEXTPROC, glProgramUniform3ivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4IVEXTPROC, glProgramUniform4ivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC, glProgramUniformMatrix2fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC, glProgramUniformMatrix3fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC, glProgramUniformMatrix4fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X3FVEXTPROC, glProgramUniformMatrix2x3fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X2FVEXTPROC, glProgramUniformMatrix3x2fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X4FVEXTPROC, glProgramUniformMatrix2x4fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X2FVEXTPROC, glProgramUniformMatrix4x2fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X4FVEXTPROC, glProgramUniformMatrix3x4fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X3FVEXTPROC, glProgramUniformMatrix4x3fvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1UIEXTPROC, glProgramUniform1uiEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2UIEXTPROC, glProgramUniform2uiEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3UIEXTPROC, glProgramUniform3uiEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4UIEXTPROC, glProgramUniform4uiEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1UIVEXTPROC, glProgramUniform1uivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2UIVEXTPROC, glProgramUniform2uivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3UIVEXTPROC, glProgramUniform3uivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4UIVEXTPROC, glProgramUniform4uivEXT, "EXT_direct_state_access", init_success);

    // texture handling ///////////////////////////////////////////////////////////////////////////
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREPARAMETERFEXTPROC, glTextureParameterfEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREPARAMETERFVEXTPROC, glTextureParameterfvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREPARAMETERIEXTPROC, glTextureParameteriEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREPARAMETERIVEXTPROC, glTextureParameterivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREPARAMETERIIVEXTPROC, glTextureParameterIivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREPARAMETERIUIVEXTPROC, glTextureParameterIuivEXT, "EXT_direct_state_access", init_success);

    SCM_INIT_GL_ENTRY(PFNGLTEXTUREIMAGE1DEXTPROC, glTextureImage1DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREIMAGE2DEXTPROC, glTextureImage2DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREIMAGE3DEXTPROC, glTextureImage3DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESUBIMAGE1DEXTPROC, glTextureSubImage1DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESUBIMAGE2DEXTPROC, glTextureSubImage2DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTURESUBIMAGE3DEXTPROC, glTextureSubImage3DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXTURESUBIMAGE1DEXTPROC, glCompressedTextureSubImage1DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXTURESUBIMAGE2DEXTPROC, glCompressedTextureSubImage2DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXTURESUBIMAGE3DEXTPROC, glCompressedTextureSubImage3DEXT, "EXT_direct_state_access", init_success);
    
    SCM_INIT_GL_ENTRY(PFNGLGETTEXTUREIMAGEEXTPROC, glGetTextureImageEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXTUREPARAMETERFVEXTPROC, glGetTextureParameterfvEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXTUREPARAMETERIVEXTPROC, glGetTextureParameterivEXT, "EXT_direct_state_access", init_success);

    SCM_INIT_GL_ENTRY(PFNGLTEXTUREBUFFEREXTPROC, glTextureBufferEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTITEXBUFFEREXTPROC, glMultiTexBufferEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDMULTITEXTUREEXTPROC, glBindMultiTextureEXT, "EXT_direct_state_access", init_success);

    // frame buffer handling //////////////////////////////////////////////////////////////////////
    SCM_INIT_GL_ENTRY(PFNGLNAMEDRENDERBUFFERSTORAGEEXTPROC, glNamedRenderbufferStorageEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDRENDERBUFFERPARAMETERIVEXTPROC, glGetNamedRenderbufferParameterivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCHECKNAMEDFRAMEBUFFERSTATUSEXTPROC, glCheckNamedFramebufferStatusEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERTEXTURE1DEXTPROC, glNamedFramebufferTexture1DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERTEXTURE2DEXTPROC, glNamedFramebufferTexture2DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERTEXTURE3DEXTPROC, glNamedFramebufferTexture3DEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERRENDERBUFFEREXTPROC, glNamedFramebufferRenderbufferEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC, glGetNamedFramebufferAttachmentParameterivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENERATETEXTUREMIPMAPEXTPROC, glGenerateTextureMipmapEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERDRAWBUFFEREXTPROC, glFramebufferDrawBufferEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERDRAWBUFFERSEXTPROC, glFramebufferDrawBuffersEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERREADBUFFEREXTPROC, glFramebufferReadBufferEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAMEBUFFERPARAMETERIVEXTPROC, glGetFramebufferParameterivEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDRENDERBUFFERSTORAGEMULTISAMPLEEXTPROC, glNamedRenderbufferStorageMultisampleEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERTEXTUREEXTPROC, glNamedFramebufferTextureEXT, "EXT_direct_state_access", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERTEXTURELAYEREXTPROC, glNamedFramebufferTextureLayerEXT, "EXT_direct_state_access", init_success);

    extension_EXT_direct_state_access_available = init_success;


    // GL_NV_bindless_texture
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLGETTEXTUREHANDLENVPROC, glGetTextureHandleNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXTURESAMPLERHANDLENVPROC, glGetTextureSamplerHandleNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKETEXTUREHANDLERESIDENTNVPROC, glMakeTextureHandleResidentNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKETEXTUREHANDLENONRESIDENTNVPROC, glMakeTextureHandleNonResidentNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETIMAGEHANDLENVPROC, glGetImageHandleNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKEIMAGEHANDLERESIDENTNVPROC, glMakeImageHandleResidentNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKEIMAGEHANDLENONRESIDENTNVPROC, glMakeImageHandleNonResidentNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMHANDLEUI64NVPROC, glUniformHandleui64NV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMHANDLEUI64VNVPROC, glUniformHandleui64vNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMHANDLEUI64NVPROC, glProgramUniformHandleui64NV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMHANDLEUI64VNVPROC, glProgramUniformHandleui64vNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISTEXTUREHANDLERESIDENTNVPROC, glIsTextureHandleResidentNV, "GL_NV_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISIMAGEHANDLERESIDENTNVPROC, glIsImageHandleResidentNV, "GL_NV_bindless_texture", init_success);
    extension_NV_bindless_texture = init_success;

    // GL_ARB_bindless_texture
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLGETTEXTUREHANDLEARBPROC, glGetTextureHandleARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXTURESAMPLERHANDLEARBPROC, glGetTextureSamplerHandleARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKETEXTUREHANDLERESIDENTARBPROC, glMakeTextureHandleResidentARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKETEXTUREHANDLENONRESIDENTARBPROC, glMakeTextureHandleNonResidentARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETIMAGEHANDLEARBPROC, glGetImageHandleARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKEIMAGEHANDLERESIDENTARBPROC, glMakeImageHandleResidentARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKEIMAGEHANDLENONRESIDENTARBPROC, glMakeImageHandleNonResidentARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMHANDLEUI64ARBPROC, glUniformHandleui64ARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMHANDLEUI64VARBPROC, glUniformHandleui64vARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMHANDLEUI64ARBPROC, glProgramUniformHandleui64ARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMHANDLEUI64VARBPROC, glProgramUniformHandleui64vARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISTEXTUREHANDLERESIDENTARBPROC, glIsTextureHandleResidentARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISIMAGEHANDLERESIDENTARBPROC, glIsImageHandleResidentARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL1UI64ARBPROC, glVertexAttribL1ui64ARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBL1UI64VARBPROC, glVertexAttribL1ui64vARB, "GL_ARB_bindless_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBLUI64VARBPROC, glGetVertexAttribLui64vARB, "GL_ARB_bindless_texture", init_success);
    extension_ARB_bindless_texture = init_success;

    // GL_NV_shader_buffer_load
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLMAKEBUFFERRESIDENTNVPROC, glMakeBufferResidentNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKEBUFFERNONRESIDENTNVPROC, glMakeBufferNonResidentNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISBUFFERRESIDENTNVPROC, glIsBufferResidentNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKENAMEDBUFFERRESIDENTNVPROC, glMakeNamedBufferResidentNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAKENAMEDBUFFERNONRESIDENTNVPROC, glMakeNamedBufferNonResidentNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLISNAMEDBUFFERRESIDENTNVPROC, glIsNamedBufferResidentNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERPARAMETERUI64VNVPROC, glGetBufferParameterui64vNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDBUFFERPARAMETERUI64VNVPROC, glGetNamedBufferParameterui64vNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGERUI64VNVPROC, glGetIntegerui64vNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMUI64NVPROC, glUniformui64NV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMUI64VNVPROC, glUniformui64vNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMUI64VNVPROC, glGetUniformui64vNV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMUI64NVPROC, glProgramUniformui64NV, "GL_NV_shader_buffer_load", init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMUI64VNVPROC, glProgramUniformui64vNV, "GL_NV_shader_buffer_load", init_success);
    extension_NV_shader_buffer_load = init_success;


    // ARB_sparse_texture
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLTEXPAGECOMMITMENTARBPROC, glTexPageCommitmentARB, "ARB_sparse_texture", init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXTUREPAGECOMMITMENTEXTPROC, glTexturePageCommitmentEXT, "ARB_sparse_texture", init_success);
    extension_ARB_sparse_texture = init_success;

    // ARB_compute_variable_group_size
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLDISPATCHCOMPUTEGROUPSIZEARBPROC, glDispatchComputeGroupSizeARB, "ARB_compute_variable_group_size", init_success);
    extension_ARB_compute_variable_group_size = init_success;

    // EXT_raster_multisample
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLRASTERSAMPLESEXTPROC, glRasterSamplesEXT, "EXT_raster_multisample", init_success);
    extension_EXT_raster_multisample = init_success;

    // NV_framebuffer_mixed_samples
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLCOVERAGEMODULATIONTABLENVPROC, glCoverageModulationTableNV, "NV_framebuffer_mixed_samples", init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETCOVERAGEMODULATIONTABLENVPROC, glGetCoverageModulationTableNV, "NV_framebuffer_mixed_samples", init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOVERAGEMODULATIONNVPROC, glCoverageModulationNV, "NV_framebuffer_mixed_samples", init_success);
    extension_NV_framebuffer_mixed_samples = init_success;

    // NV_fragment_coverage_to_color
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLFRAGMENTCOVERAGECOLORNVPROC, glFragmentCoverageColorNV, "NV_fragment_coverage_to_color", init_success);
    extension_NV_fragment_coverage_to_color = init_success;

    // NV_sample_locations
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERSAMPLELOCATIONSFVNVPROC, glFramebufferSampleLocationsfvNV, "NV_sample_locations", init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDFRAMEBUFFERSAMPLELOCATIONSFVNVPROC, glNamedFramebufferSampleLocationsfvNV, "NV_sample_locations", init_success);
    SCM_INIT_GL_ENTRY(PFNGLRESOLVEDEPTHVALUESNVPROC, glResolveDepthValuesNV, "NV_sample_locations", init_success);
    extension_NV_sample_locations = init_success;

    // NV_conservative_raster
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLSUBPIXELPRECISIONBIASNVPROC, glSubpixelPrecisionBiasNV, "NV_conservative_raster", init_success);
    extension_NV_conservative_raster = init_success;

    glout() << log::outdent;
    glout() << log::info << "finished initializing function entry points..." << log::end;

#undef SCM_INIT_GL_ENTRY
}

} // namespace opengl
} // namespace gl
} // namespace scm
