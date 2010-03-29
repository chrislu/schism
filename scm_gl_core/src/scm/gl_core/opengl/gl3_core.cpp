
#include "gl3_core.h"

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
#include <scm/gl_core/opengl/config.h>

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
gl3_core::is_initialized() const
{
    return (_initialized);
}

bool
gl3_core::is_supported(const std::string& ext) const
{
    if (_extensions.find(ext) != _extensions.end()) {
        return (true);
    }
    else {
        return (false);
    }
}

const gl3_core::context_info&
gl3_core::context_information() const
{
    return (_context_info);
}

gl3_core::gl3_core()
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

    extension_EXT_direct_state_access_available = false;
}

bool
gl3_core::initialize()
{
    if (is_initialized()) {
        return (true);
    }
    log::logger_format_saver save_indent(glout());
    glout() << log::info << "gl3_core::initialize(): starting to initialize gl core:" << log::end;
    glout() << log::indent;

    init_entry_points();

    if (!version_1_1_available) {
        glerr() << log::fatal << "gl3_core::initialize(): unable to initialize gl core, missing vital entry points" << log::end;
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
                    << "gl3_core::initialize(): "
                    << "unable to parse OpenGL Version string, malformed version string ('"
                    << gl_version_string << "')" << log::end;
            return (false);
        }
    }

    _context_info._vendor.assign(reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
    _context_info._renderer.assign(reinterpret_cast<const char*>(glGetString(GL_RENDERER)));

    if (_context_info._version_major == 3) {
        if (    _context_info._version_minor == 0 && !version_3_0_available) {
            glout() << log::warning << "gl3_core::initialize(): OpenGL version 3.0 reported but missing entry points detected" << log::end;
        }
        if (    _context_info._version_minor == 1 && !version_3_1_available) {
            glout() << log::warning << "gl3_core::initialize(): OpenGL version 3.1 reported but missing entry points detected" << log::end;
        }
        if (    _context_info._version_minor == 2 && !version_3_2_available) {
            glout() << log::warning << "gl3_core::initialize(): OpenGL version 3.2 reported but missing entry points detected" << log::end;
        }
        if (    _context_info._version_minor == 3 && !version_3_3_available) {
            glout() << log::warning << "gl3_core::initialize(): OpenGL version 3.3 reported but missing entry points detected" << log::end;
        }
    }
    else if (_context_info._version_major == 4) {
        if (    _context_info._version_minor == 0 && !version_4_0_available) {
            glout() << log::warning << "gl3_core::initialize(): OpenGL version 4.0 reported but missing entry points detected" << log::end;
        }
    }
    else if (_context_info._version_major < 3) {
        glerr() << log::fatal << "gl3_core::initialize(): at least OpenGL version 3.0 requiered" << log::end;
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
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    if (!is_supported("GL_EXT_direct_state_access")) {
        glout() << log::warning
                << "gl3_core::initialize(): "
                << "GL_EXT_direct_state_access not supported but enabled for scm_gl_core use "
                << "(undefine SCM_GL_CORE_USE_DIRECT_STATE_ACCESS!)" << log::end;
    }
#endif // SCM_GL_CORE_USE_DIRECT_STATE_ACCESS

    glout() << log::outdent;
    glout() << log::info << "gl3_core::initialize(): finished to initializing gl core." << log::end;

    return (true);
}

std::ostream& operator<<(std::ostream& out_stream, const gl3_core& c)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_stream << "vendor:      " << c._context_info._vendor << std::endl
               << "renderer:    " << c._context_info._renderer << std::endl
               << "version:     " << c._context_info._version_major << "." 
                                  << c._context_info._version_minor << "." 
                                  << c._context_info._version_release;
    if (!c._context_info._version_info.empty())
         out_stream               << " " << c._context_info._version_info;
    if (!c._context_info._profile_string.empty())
         out_stream               << " " << c._context_info._profile_string;
    out_stream << std::endl;
    out_stream << "extensions : " << "(found " << c._extensions.size() << ")" << std::endl;

    for (gl3_core::string_set::const_iterator i = c._extensions.begin(); i != c._extensions.end(); ++i) {
        out_stream << "             " << *i << std::endl;
    }

    return (out_stream);
}

void
gl3_core::init_entry_points()
{
    /* Visual Studio regex find and replace
        find:           {PFN[^ ]+}[ ]+{gl[^;]+}
        replace with:   \2 = gl_proc_address<\1>("\2")
                        SCM_INIT_GL_ENTRY(\1, \2, init_success)
    */

#define SCM_INIT_GL_ENTRY(PFN, fun, errflag)                                                \
    if (0 == (fun = gl_proc_address<PFN>(#fun))) {                                          \
        errflag = false;                                                                    \
        glout() << log::warning << "missing entry point " << #fun << log::end;              \
    }

    glout() << log::info << "initializing function entry points..." << log::end;

    bool init_success = true;

    // version 1.0 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLCULLFACEPROC, glCullFace, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRONTFACEPROC, glFrontFace, init_success);
    SCM_INIT_GL_ENTRY(PFNGLHINTPROC, glHint, init_success);
    SCM_INIT_GL_ENTRY(PFNGLLINEWIDTHPROC, glLineWidth, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTSIZEPROC, glPointSize, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOLYGONMODEPROC, glPolygonMode, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSCISSORPROC, glScissor, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERFPROC, glTexParameterf, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERFVPROC, glTexParameterfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIPROC, glTexParameteri, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIVPROC, glTexParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE1DPROC, glTexImage1D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE2DPROC, glTexImage2D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWBUFFERPROC, glDrawBuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARPROC, glClear, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARCOLORPROC, glClearColor, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARSTENCILPROC, glClearStencil, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARDEPTHPROC, glClearDepth, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILMASKPROC, glStencilMask, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOLORMASKPROC, glColorMask, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHMASKPROC, glDepthMask, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEPROC, glDisable, init_success);
    SCM_INIT_GL_ENTRY(PFNGLENABLEPROC, glEnable, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFINISHPROC, glFinish, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFLUSHPROC, glFlush, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLENDFUNCPROC, glBlendFunc, init_success);
    SCM_INIT_GL_ENTRY(PFNGLLOGICOPPROC, glLogicOp, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILFUNCPROC, glStencilFunc, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILOPPROC, glStencilOp, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHFUNCPROC, glDepthFunc, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPIXELSTOREFPROC, glPixelStoref, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPIXELSTOREIPROC, glPixelStorei, init_success);
    SCM_INIT_GL_ENTRY(PFNGLREADBUFFERPROC, glReadBuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLREADPIXELSPROC, glReadPixels, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBOOLEANVPROC, glGetBooleanv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETDOUBLEVPROC, glGetDoublev, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETERRORPROC, glGetError, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFLOATVPROC, glGetFloatv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGERVPROC, glGetIntegerv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSTRINGPROC, glGetString, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXIMAGEPROC, glGetTexImage, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERFVPROC, glGetTexParameterfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERIVPROC, glGetTexParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXLEVELPARAMETERFVPROC, glGetTexLevelParameterfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXLEVELPARAMETERIVPROC, glGetTexLevelParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISENABLEDPROC, glIsEnabled, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDEPTHRANGEPROC, glDepthRange, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVIEWPORTPROC, glViewport, init_success);
    version_1_0_available = init_success;
                                            
    // version 1.1 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLDRAWARRAYSPROC, glDrawArrays, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSPROC, glDrawElements, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPOINTERVPROC, glGetPointerv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOLYGONOFFSETPROC, glPolygonOffset, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXIMAGE1DPROC, glCopyTexImage1D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXIMAGE2DPROC, glCopyTexImage2D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXSUBIMAGE1DPROC, glCopyTexSubImage1D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXSUBIMAGE2DPROC, glCopyTexSubImage2D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSUBIMAGE1DPROC, glTexSubImage1D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSUBIMAGE2DPROC, glTexSubImage2D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDTEXTUREPROC, glBindTexture, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETETEXTURESPROC, glDeleteTextures, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENTEXTURESPROC, glGenTextures, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISTEXTUREPROC, glIsTexture, init_success);
    version_1_1_available = version_1_0_available && init_success;
                                            
    // version 1.2 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBLENDCOLORPROC, glBlendColor, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLENDEQUATIONPROC, glBlendEquation, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWRANGEELEMENTSPROC, glDrawRangeElements, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE3DPROC, glTexImage3D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXSUBIMAGE3DPROC, glTexSubImage3D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOPYTEXSUBIMAGE3DPROC, glCopyTexSubImage3D, init_success);
    version_1_2_available = version_1_1_available && init_success;
                                            
    // version 1.3 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLACTIVETEXTUREPROC, glActiveTexture, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLECOVERAGEPROC, glSampleCoverage, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXIMAGE3DPROC, glCompressedTexImage3D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXIMAGE2DPROC, glCompressedTexImage2D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXIMAGE1DPROC, glCompressedTexImage1D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXSUBIMAGE3DPROC, glCompressedTexSubImage3D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXSUBIMAGE2DPROC, glCompressedTexSubImage2D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPRESSEDTEXSUBIMAGE1DPROC, glCompressedTexSubImage1D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETCOMPRESSEDTEXIMAGEPROC, glGetCompressedTexImage, init_success);
    version_1_3_available = version_1_2_available && init_success;
                                            
    // version 1.4 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBLENDFUNCSEPARATEPROC, glBlendFuncSeparate, init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWARRAYSPROC, glMultiDrawArrays, init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWELEMENTSPROC, glMultiDrawElements, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERFPROC, glPointParameterf, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERFVPROC, glPointParameterfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERIPROC, glPointParameteri, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPOINTPARAMETERIVPROC, glPointParameteriv, init_success);
    version_1_4_available = version_1_3_available && init_success;
                                            
    // version 1.5 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLGENQUERIESPROC, glGenQueries, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEQUERIESPROC, glDeleteQueries, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISQUERYPROC, glIsQuery, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBEGINQUERYPROC, glBeginQuery, init_success);
    SCM_INIT_GL_ENTRY(PFNGLENDQUERYPROC, glEndQuery, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYIVPROC, glGetQueryiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTIVPROC, glGetQueryObjectiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTUIVPROC, glGetQueryObjectuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERPROC, glBindBuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEBUFFERSPROC, glDeleteBuffers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENBUFFERSPROC, glGenBuffers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISBUFFERPROC, glIsBuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBUFFERDATAPROC, glBufferData, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBUFFERSUBDATAPROC, glBufferSubData, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERSUBDATAPROC, glGetBufferSubData, init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAPBUFFERPROC, glMapBuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNMAPBUFFERPROC, glUnmapBuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERPARAMETERIVPROC, glGetBufferParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERPOINTERVPROC, glGetBufferPointerv, init_success);
    version_1_5_available = version_1_4_available && init_success;

    // version 2.0 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLBLENDEQUATIONSEPARATEPROC, glBlendEquationSeparate, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWBUFFERSPROC, glDrawBuffers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILOPSEPARATEPROC, glStencilOpSeparate, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILFUNCSEPARATEPROC, glStencilFuncSeparate, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSTENCILMASKSEPARATEPROC, glStencilMaskSeparate, init_success);
    SCM_INIT_GL_ENTRY(PFNGLATTACHSHADERPROC, glAttachShader, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDATTRIBLOCATIONPROC, glBindAttribLocation, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCOMPILESHADERPROC, glCompileShader, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCREATEPROGRAMPROC, glCreateProgram, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCREATESHADERPROC, glCreateShader, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEPROGRAMPROC, glDeleteProgram, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETESHADERPROC, glDeleteShader, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDETACHSHADERPROC, glDetachShader, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEVERTEXATTRIBARRAYPROC, glDisableVertexAttribArray, init_success);
    SCM_INIT_GL_ENTRY(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEATTRIBPROC, glGetActiveAttrib, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMPROC, glGetActiveUniform, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETATTACHEDSHADERSPROC, glGetAttachedShaders, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETATTRIBLOCATIONPROC, glGetAttribLocation, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMIVPROC, glGetProgramiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSHADERIVPROC, glGetShaderiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSHADERSOURCEPROC, glGetShaderSource, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMFVPROC, glGetUniformfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMIVPROC, glGetUniformiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBDVPROC, glGetVertexAttribdv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBFVPROC, glGetVertexAttribfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBIVPROC, glGetVertexAttribiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBPOINTERVPROC, glGetVertexAttribPointerv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISPROGRAMPROC, glIsProgram, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISSHADERPROC, glIsShader, init_success);
    SCM_INIT_GL_ENTRY(PFNGLLINKPROGRAMPROC, glLinkProgram, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSHADERSOURCEPROC, glShaderSource, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUSEPROGRAMPROC, glUseProgram, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1FPROC, glUniform1f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2FPROC, glUniform2f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3FPROC, glUniform3f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4FPROC, glUniform4f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1IPROC, glUniform1i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2IPROC, glUniform2i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3IPROC, glUniform3i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4IPROC, glUniform4i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1FVPROC, glUniform1fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2FVPROC, glUniform2fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3FVPROC, glUniform3fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4FVPROC, glUniform4fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1IVPROC, glUniform1iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2IVPROC, glUniform2iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3IVPROC, glUniform3iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4IVPROC, glUniform4iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2FVPROC, glUniformMatrix2fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3FVPROC, glUniformMatrix3fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4FVPROC, glUniformMatrix4fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVALIDATEPROGRAMPROC, glValidateProgram, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1DPROC, glVertexAttrib1d, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1DVPROC, glVertexAttrib1dv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1FPROC, glVertexAttrib1f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1FVPROC, glVertexAttrib1fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1SPROC, glVertexAttrib1s, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB1SVPROC, glVertexAttrib1sv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2DPROC, glVertexAttrib2d, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2DVPROC, glVertexAttrib2dv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2FPROC, glVertexAttrib2f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2FVPROC, glVertexAttrib2fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2SPROC, glVertexAttrib2s, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB2SVPROC, glVertexAttrib2sv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3DPROC, glVertexAttrib3d, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3DVPROC, glVertexAttrib3dv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3FPROC, glVertexAttrib3f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3FVPROC, glVertexAttrib3fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3SPROC, glVertexAttrib3s, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB3SVPROC, glVertexAttrib3sv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NBVPROC, glVertexAttrib4Nbv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NIVPROC, glVertexAttrib4Niv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NSVPROC, glVertexAttrib4Nsv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUBPROC, glVertexAttrib4Nub, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUBVPROC, glVertexAttrib4Nubv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUIVPROC, glVertexAttrib4Nuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4NUSVPROC, glVertexAttrib4Nusv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4BVPROC, glVertexAttrib4bv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4DPROC, glVertexAttrib4d, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4DVPROC, glVertexAttrib4dv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4FPROC, glVertexAttrib4f, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4FVPROC, glVertexAttrib4fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4IVPROC, glVertexAttrib4iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4SPROC, glVertexAttrib4s, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4SVPROC, glVertexAttrib4sv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4UBVPROC, glVertexAttrib4ubv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4UIVPROC, glVertexAttrib4uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIB4USVPROC, glVertexAttrib4usv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer, init_success);
    version_2_0_available = version_1_5_available && init_success;
                             
    // version 2.1 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2X3FVPROC, glUniformMatrix2x3fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3X2FVPROC, glUniformMatrix3x2fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX2X4FVPROC, glUniformMatrix2x4fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4X2FVPROC, glUniformMatrix4x2fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX3X4FVPROC, glUniformMatrix3x4fv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMMATRIX4X3FVPROC, glUniformMatrix4x3fv, init_success);
    version_2_1_available = version_2_0_available && init_success;

    // version 3.0 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLCOLORMASKIPROC, glColorMaski, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBOOLEANI_VPROC, glGetBooleani_v, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGERI_VPROC, glGetIntegeri_v, init_success);
    SCM_INIT_GL_ENTRY(PFNGLENABLEIPROC, glEnablei, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDISABLEIPROC, glDisablei, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISENABLEDIPROC, glIsEnabledi, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBEGINTRANSFORMFEEDBACKPROC, glBeginTransformFeedback, init_success);
    SCM_INIT_GL_ENTRY(PFNGLENDTRANSFORMFEEDBACKPROC, glEndTransformFeedback, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERRANGEPROC, glBindBufferRange, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDBUFFERBASEPROC, glBindBufferBase, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTRANSFORMFEEDBACKVARYINGSPROC, glTransformFeedbackVaryings, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTRANSFORMFEEDBACKVARYINGPROC, glGetTransformFeedbackVarying, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLAMPCOLORPROC, glClampColor, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBEGINCONDITIONALRENDERPROC, glBeginConditionalRender, init_success);
    SCM_INIT_GL_ENTRY(PFNGLENDCONDITIONALRENDERPROC, glEndConditionalRender, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBIPOINTERPROC, glVertexAttribIPointer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBIIVPROC, glGetVertexAttribIiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETVERTEXATTRIBIUIVPROC, glGetVertexAttribIuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1IPROC, glVertexAttribI1i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2IPROC, glVertexAttribI2i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3IPROC, glVertexAttribI3i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4IPROC, glVertexAttribI4i, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1UIPROC, glVertexAttribI1ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2UIPROC, glVertexAttribI2ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3UIPROC, glVertexAttribI3ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4UIPROC, glVertexAttribI4ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1IVPROC, glVertexAttribI1iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2IVPROC, glVertexAttribI2iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3IVPROC, glVertexAttribI3iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4IVPROC, glVertexAttribI4iv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI1UIVPROC, glVertexAttribI1uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI2UIVPROC, glVertexAttribI2uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI3UIVPROC, glVertexAttribI3uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4UIVPROC, glVertexAttribI4uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4BVPROC, glVertexAttribI4bv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4SVPROC, glVertexAttribI4sv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4UBVPROC, glVertexAttribI4ubv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLVERTEXATTRIBI4USVPROC, glVertexAttribI4usv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMUIVPROC, glGetUniformuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDFRAGDATALOCATIONPROC, glBindFragDataLocation, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAGDATALOCATIONPROC, glGetFragDataLocation, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1UIPROC, glUniform1ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2UIPROC, glUniform2ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3UIPROC, glUniform3ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4UIPROC, glUniform4ui, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM1UIVPROC, glUniform1uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM2UIVPROC, glUniform2uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM3UIVPROC, glUniform3uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORM4UIVPROC, glUniform4uiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIIVPROC, glTexParameterIiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXPARAMETERIUIVPROC, glTexParameterIuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERIIVPROC, glGetTexParameterIiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETTEXPARAMETERIUIVPROC, glGetTexParameterIuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERIVPROC, glClearBufferiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERUIVPROC, glClearBufferuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERFVPROC, glClearBufferfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLEARBUFFERFIPROC, glClearBufferfi, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSTRINGIPROC, glGetStringi, init_success);
    // use ARB_framebuffer_object
    SCM_INIT_GL_ENTRY(PFNGLISRENDERBUFFERPROC, glIsRenderbuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDRENDERBUFFERPROC, glBindRenderbuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETERENDERBUFFERSPROC, glDeleteRenderbuffers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENRENDERBUFFERSPROC, glGenRenderbuffers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLRENDERBUFFERSTORAGEPROC, glRenderbufferStorage, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETRENDERBUFFERPARAMETERIVPROC, glGetRenderbufferParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISFRAMEBUFFERPROC, glIsFramebuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURE1DPROC, glFramebufferTexture1D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURE3DPROC, glFramebufferTexture3D, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERRENDERBUFFERPROC, glFramebufferRenderbuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVPROC, glGetFramebufferAttachmentParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENERATEMIPMAPPROC, glGenerateMipmap, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBLITFRAMEBUFFERPROC, glBlitFramebuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLRENDERBUFFERSTORAGEMULTISAMPLEPROC, glRenderbufferStorageMultisample, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTURELAYERPROC, glFramebufferTextureLayer, init_success);
    // use ARB_map_buffer_ranger
    SCM_INIT_GL_ENTRY(PFNGLMAPBUFFERRANGEPROC, glMapBufferRange, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFLUSHMAPPEDBUFFERRANGEPROC, glFlushMappedBufferRange, init_success);
    // use ARB_vertex_array_object          
    SCM_INIT_GL_ENTRY(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISVERTEXARRAYPROC, glIsVertexArray, init_success);
    version_3_0_available = version_2_1_available && init_success;

    // version 3.1 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLDRAWARRAYSINSTANCEDPROC, glDrawArraysInstanced, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSINSTANCEDPROC, glDrawElementsInstanced, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXBUFFERPROC, glTexBuffer, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPRIMITIVERESTARTINDEXPROC, glPrimitiveRestartIndex, init_success);
    // use ARB_copy_buffer                  
    SCM_INIT_GL_ENTRY(PFNGLCOPYBUFFERSUBDATAPROC, glCopyBufferSubData, init_success);
    // use ARB_uniform_buffer_object        
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMINDICESPROC, glGetUniformIndices, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMSIVPROC, glGetActiveUniformsiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMNAMEPROC, glGetActiveUniformName, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETUNIFORMBLOCKINDEXPROC, glGetUniformBlockIndex, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMBLOCKIVPROC, glGetActiveUniformBlockiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETACTIVEUNIFORMBLOCKNAMEPROC, glGetActiveUniformBlockName, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNIFORMBLOCKBINDINGPROC, glUniformBlockBinding, init_success);
    version_3_1_available = version_3_0_available && init_success;

    // version 3.2 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGER64I_VPROC, glGetInteger64i_v, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETBUFFERPARAMETERI64VPROC, glGetBufferParameteri64v, init_success);
    //SCM_INIT_GL_ENTRY(PFNGLPROGRAMPARAMETERIPROC, glProgramParameteri, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTUREPROC, glFramebufferTexture, init_success);
    //SCM_INIT_GL_ENTRY(PFNGLFRAMEBUFFERTEXTUREFACEPROC, glFramebufferTextureFace, init_success);
    // use ARB_draw_elements_base_vertex    
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSBASEVERTEXPROC, glDrawElementsBaseVertex, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWRANGEELEMENTSBASEVERTEXPROC, glDrawRangeElementsBaseVertex, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDRAWELEMENTSINSTANCEDBASEVERTEXPROC, glDrawElementsInstancedBaseVertex, init_success);
    SCM_INIT_GL_ENTRY(PFNGLMULTIDRAWELEMENTSBASEVERTEXPROC, glMultiDrawElementsBaseVertex, init_success);
    // use ARB_provoking_vertex             
    SCM_INIT_GL_ENTRY(PFNGLPROVOKINGVERTEXPROC, glProvokingVertex, init_success);
    // use ARB_sync                         
    SCM_INIT_GL_ENTRY(PFNGLFENCESYNCPROC, glFenceSync, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISSYNCPROC, glIsSync, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETESYNCPROC, glDeleteSync, init_success);
    SCM_INIT_GL_ENTRY(PFNGLCLIENTWAITSYNCPROC, glClientWaitSync, init_success);
    SCM_INIT_GL_ENTRY(PFNGLWAITSYNCPROC, glWaitSync, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETINTEGER64VPROC, glGetInteger64v, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSYNCIVPROC, glGetSynciv, init_success);
    // use ARB_texture_multisample          
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE2DMULTISAMPLEPROC, glTexImage2DMultisample, init_success);
    SCM_INIT_GL_ENTRY(PFNGLTEXIMAGE3DMULTISAMPLEPROC, glTexImage3DMultisample, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETMULTISAMPLEFVPROC, glGetMultisamplefv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLEMASKIPROC, glSampleMaski, init_success);
    version_3_2_available = version_3_1_available && init_success;

    // version 3.3 ////////////////////////////////////////////////////////////////////////////////
    init_success = true;
    // use GL_ARB_shader_bit_encoding
    // use ARB_blend_func_extended
    SCM_INIT_GL_ENTRY(PFNGLBINDFRAGDATALOCATIONINDEXEDPROC, glBindFragDataLocationIndexed, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETFRAGDATAINDEXPROC, glGetFragDataIndex, init_success);
    // use GL_ARB_explicit_attrib_location
    // use GL_ARB_occlusion_query2
    // use ARB_sampler_objects
    SCM_INIT_GL_ENTRY(PFNGLGENSAMPLERSPROC, glGenSamplers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLDELETESAMPLERSPROC, glDeleteSamplers, init_success);
    SCM_INIT_GL_ENTRY(PFNGLISSAMPLERPROC, glIsSampler, init_success);
    SCM_INIT_GL_ENTRY(PFNGLBINDSAMPLERPROC, glBindSampler, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIPROC, glSamplerParameteri, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIVPROC, glSamplerParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERFPROC, glSamplerParameterf, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERFVPROC, glSamplerParameterfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIIVPROC, glSamplerParameterIiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLSAMPLERPARAMETERIUIVPROC, glSamplerParameterIuiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERIVPROC, glGetSamplerParameteriv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERIIVPROC, glGetSamplerParameterIiv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERFVPROC, glGetSamplerParameterfv, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETSAMPLERPARAMETERIFVPROC, glGetSamplerParameterIfv, init_success);
    // use GL_ARB_texture_rgb10_a2ui
    // use GL_ARB_texture_swizzle
    // use ARB_timer_query
    SCM_INIT_GL_ENTRY(PFNGLQUERYCOUNTERPROC, glQueryCounter, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTI64VPROC, glGetQueryObjecti64v, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETQUERYOBJECTUI64VPROC, glGetQueryObjectui64v, init_success);
    // use GL_ARB_texture_swizzle
    /// TODO missing entry points
    // use ARB_vertex_type_2_10_10_10_rev
    // non which concern core profile
    version_3_3_available = version_3_2_available && init_success;

    // EXT_direct_state_access
    // buffer handling ////////////////////////////////////////////////////////////////////////////
    init_success = true;
    bool    extension_EXT_direct_state_access_available;
    SCM_INIT_GL_ENTRY(PFNGLNAMEDBUFFERDATAEXTPROC, glNamedBufferDataEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLNAMEDBUFFERSUBDATAEXTPROC, glNamedBufferSubDataEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLMAPNAMEDBUFFEREXTPROC, glMapNamedBufferEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLUNMAPNAMEDBUFFEREXTPROC, glUnmapNamedBufferEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDBUFFERPARAMETERIVEXTPROC, glGetNamedBufferParameterivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDBUFFERPOINTERVEXTPROC, glGetNamedBufferPointervEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLGETNAMEDBUFFERSUBDATAEXTPROC, glGetNamedBufferSubDataEXT, init_success);

    SCM_INIT_GL_ENTRY(PFNGLMAPNAMEDBUFFERRANGEEXTPROC, glMapNamedBufferRangeEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLFLUSHMAPPEDNAMEDBUFFERRANGEEXTPROC, glFlushMappedNamedBufferRangeEXT, init_success);

    // shader handling ////////////////////////////////////////////////////////////////////////////
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1FEXTPROC, glProgramUniform1fEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2FEXTPROC, glProgramUniform2fEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3FEXTPROC, glProgramUniform3fEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4FEXTPROC, glProgramUniform4fEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1IEXTPROC, glProgramUniform1iEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2IEXTPROC, glProgramUniform2iEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3IEXTPROC, glProgramUniform3iEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4IEXTPROC, glProgramUniform4iEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1FVEXTPROC, glProgramUniform1fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2FVEXTPROC, glProgramUniform2fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3FVEXTPROC, glProgramUniform3fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4FVEXTPROC, glProgramUniform4fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1IVEXTPROC, glProgramUniform1ivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2IVEXTPROC, glProgramUniform2ivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3IVEXTPROC, glProgramUniform3ivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4IVEXTPROC, glProgramUniform4ivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2FVEXTPROC, glProgramUniformMatrix2fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3FVEXTPROC, glProgramUniformMatrix3fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4FVEXTPROC, glProgramUniformMatrix4fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X3FVEXTPROC, glProgramUniformMatrix2x3fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X2FVEXTPROC, glProgramUniformMatrix3x2fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX2X4FVEXTPROC, glProgramUniformMatrix2x4fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X2FVEXTPROC, glProgramUniformMatrix4x2fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX3X4FVEXTPROC, glProgramUniformMatrix3x4fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORMMATRIX4X3FVEXTPROC, glProgramUniformMatrix4x3fvEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1UIEXTPROC, glProgramUniform1uiEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2UIEXTPROC, glProgramUniform2uiEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3UIEXTPROC, glProgramUniform3uiEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4UIEXTPROC, glProgramUniform4uiEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM1UIVEXTPROC, glProgramUniform1uivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM2UIVEXTPROC, glProgramUniform2uivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM3UIVEXTPROC, glProgramUniform3uivEXT, init_success);
    SCM_INIT_GL_ENTRY(PFNGLPROGRAMUNIFORM4UIVEXTPROC, glProgramUniform4uivEXT, init_success);
    extension_EXT_direct_state_access_available = init_success;

    glout() << log::info << "finished initializing function entry points..." << log::end;

#undef SCM_INIT_GL_ENTRY
}

} // namespace opengl
} // namespace gl
} // namespace scm
