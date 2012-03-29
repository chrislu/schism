
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "gl_module.h"

#include <cassert>
#include <ostream>

#include <boost/bind.hpp>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/tokenizer.hpp>

#include <scm/core.h>
#include <scm/core/module/initializer.h>
#include <scm/gl_classic/opengl.h>

SCM_SINGLETON_PLACEMENT(gl_core, scm::gl_classic::gl_module)

namespace {

static void init_module()
{
    scm::module::initializer::add_post_core_init_function(      boost::bind(&scm::gl_classic::gl_module::initialize, &scm::gl_classic::ogl::get(), _1));
    scm::module::initializer::add_pre_core_shutdown_function(   boost::bind(&scm::gl_classic::gl_module::shutdown,   &scm::gl_classic::ogl::get(), _1));
}

static scm::module::static_initializer  static_initialize(init_module);

} // namespace

namespace scm {
namespace gl_classic {

gl_module::gl_module()
  : _initialized(false)
{
}

gl_module::~gl_module()
{
}

bool
gl_module::initialize(core& c)
{
    if (_initialized) {
        scm::err() << log::warning
                   << "gl_module::initialize(): "
                   << "allready initialized" << log::end;
        return (true);
    }

    scm::out() << log::info
               << "initializing scm.gl module:"  << log::end;

    scm::out() << log::info
               << "successfully initialized scm::ogl library"  << log::end;

    _initialized = true;
    return (true);
}

bool
gl_module::initialize_gl_context()
{
    scm::out() << log::info
               << "initializing scm.gl module context:"  << log::end;


    // enable the checking for GL3 'extensions'
    glewExperimental = GL_TRUE;

    if (glewInit() != GLEW_OK) {
        scm::err() << log::error
                   << "gl_module::initialize_gl_context(): "
                   << "unable to initialize OpenGL sub system (GLEW Init failed)" << log::end;

        return (false);
    }

    // parse the version string
    const char* gl_version_string = reinterpret_cast<const char*>(glGetString(GL_VERSION));

    namespace bsc = boost::spirit::classic;

    bsc::rule<> gl_version_string_format =      bsc::int_p[bsc::assign_a(_context_info._version_major)]
                                           >>   bsc::ch_p('.')
                                           >>   bsc::int_p[bsc::assign_a(_context_info._version_minor)]
                                           >> !(bsc::ch_p('.') >> bsc::int_p[bsc::assign_a(_context_info._version_release)])
                                           >> (*bsc::anychar_p)[bsc::assign_a(_context_info._version_info)];

    if (!bsc::parse(gl_version_string, gl_version_string_format, bsc::space_p).full) {
        scm::err() << log::error
                   << "gl_module::initialize_gl_context(): "
                   << "unable to parse OpenGL Version string, malformed version string ('"
                   << gl_version_string << "')" << log::end;

        return (false);
    }

    _context_info._vendor   = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    _context_info._renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));

    // get the extension strings
    if (_context_info._version_major >= 3) {
        GLint num_extensions = 0;

        glGetIntegerv(GL_NUM_EXTENSIONS, &num_extensions);
        for (int i = 0; i < num_extensions; ++i) {
            const char* extension_string = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));

            _supported_extensions.insert(std::string(extension_string));
        }

        int profile_mask = 0;
#ifdef GL_CONTEXT_PROFILE_MASK
        glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile_mask);
        if (profile_mask & GL_CONTEXT_CORE_PROFILE_BIT) {
            _context_info._profile.assign("core profile");
        }
        else if (profile_mask & GL_CONTEXT_COMPATIBILITY_PROFILE_BIT) {
            _context_info._profile.assign("compatibility profile");
        }
        else
#endif
        {
            _context_info._profile.assign("unknown profile");
        }
    }
    else {
        std::string gl_ext_string = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));

        typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
        boost::char_separator<char> space_separator(" ");
        tokenizer                   extension_strings(gl_ext_string, space_separator);

        for (tokenizer::const_iterator i = extension_strings.begin(); i != extension_strings.end(); ++i) {
            _supported_extensions.insert(std::string(*i));
        }
    }

#ifdef SCM_GL_USE_DIRECT_STATE_ACCESS
    if (is_supported("GL_EXT_direct_state_access")) {
        scm::out() << log::info
                   << "gl_module::initialize_gl_context(): "
                   << "GL_EXT_direct_state_access supported and enabled for scm_gl use."  << log::end;
    }
    else {
        scm::err() << log::error
                   << "gl_module::initialize_gl_context(): "
                   << "GL_EXT_direct_state_access not supported but enabled for scm_gl use "
                   << "(undefine SCM_GL_USE_DIRECT_STATE_ACCESS!)" << log::end;
        return (false);
    }
#endif // SCM_GL_USE_DIRECT_STATE_ACCESS

    scm::out() << log::info
               << "successfully initialized scm.gl module context:"  << log::end;

    return (true);
}

bool
gl_module::shutdown(core& c)
{
    scm::out() << log::info
               << "shutting down scm.gl module:"  << log::end;

    _initialized = false;

    scm::out() << log::info
               << "successfully shut down scm.gl module"  << log::end;

    return (true);
}

const
gl_module::context_info&
gl_module::context_information() const
{
    return (_context_info);
}

bool
gl_module::is_supported(const std::string& ext) const
{
    if (_supported_extensions.find(ext) != _supported_extensions.end()) {
        return (true);
    }
    else {
        return (false);
    }
    //return (glewIsSupported(ext.c_str()) == GL_TRUE ? true : false);
}

std::ostream& operator<<(std::ostream& out_stream, const gl_module& gl_mod)
{
    std::ostream::sentry const  out_sentry(out_stream);

    out_stream << "vendor:      " << gl_mod.context_information()._vendor << std::endl
               << "renderer:    " << gl_mod.context_information()._renderer << std::endl
               << "version:     " << gl_mod.context_information()._version_major << "." 
                                  << gl_mod.context_information()._version_minor << "." 
                                  << gl_mod.context_information()._version_release << " " 
                                  << gl_mod.context_information()._version_info << " "
                                  << gl_mod.context_information()._profile << std::endl
               << "extensions : " << "(found " << gl_mod._supported_extensions.size() << ")" << std::endl;

    for (gl_module::extension_string_container::const_iterator i = gl_mod._supported_extensions.begin(); i != gl_mod._supported_extensions.end(); ++i) {
        out_stream << "             " << *i << std::endl;
    }

    return (out_stream);
}

} // namespace gl_classic
} // namespace scm
