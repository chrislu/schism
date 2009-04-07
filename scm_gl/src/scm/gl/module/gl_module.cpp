
#include "gl_module.h"

#include <cassert>
#include <ostream>

#include <boost/bind.hpp>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/tokenizer.hpp>

#include <scm/core.h>
#include <scm/core/module/initializer.h>
#include <scm/gl/opengl.h>

namespace {

static void init_module()
{
    scm::module::initializer::add_post_core_init_function(      boost::bind(&scm::gl::gl_module::initialize, &scm::gl::ogl::get(), _1));
    scm::module::initializer::add_pre_core_shutdown_function(   boost::bind(&scm::gl::gl_module::shutdown,   &scm::gl::ogl::get(), _1));
}

static scm::module::static_initializer  static_initialize(init_module);

} // namespace

SCM_SINGLETON_PLACEMENT(ogl, scm::gl::gl_module)

namespace scm {
namespace gl {

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
        scm::err() << scm::log_level(scm::logging::ll_warning)
                   << "gl_module::initialize()(): "
                   << "allready initialized" << std::endl;
        return (true);
    }

    scm::out() << scm::log_level(scm::logging::ll_info)
               << "initializing scm.gl module:"  << std::endl;

    scm::out() << scm::log_level(scm::logging::ll_info)
               << "successfully initialized scm::ogl library"  << std::endl;

    _initialized = true;
    return (true);
}

bool
gl_module::initialize_gl_context()
{
    scm::out() << scm::log_level(scm::logging::ll_info)
               << "initializing scm.gl module context:"  << std::endl;


    // enable the checking for GL3 'extensions'
    glewExperimental = GL_TRUE;

    if (glewInit() != GLEW_OK) {
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "gl_module::initialize_gl_context(): "
                   << "unable to initialize OpenGL sub system (GLEW Init failed)" << std::endl;

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
        scm::err() << scm::log_level(scm::logging::ll_error)
                   << "gl_module::initialize_gl_context(): "
                   << "unable to parse OpenGL Version string, malformed version string ('"
                   << gl_version_string << "')" << std::endl;

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

    scm::out() << scm::log_level(scm::logging::ll_info)
               << "successfully initialized scm.gl module context:"  << std::endl;

    return (true);
}

bool
gl_module::shutdown(core& c)
{
    scm::out() << scm::log_level(scm::logging::ll_info)
               << "shutting down scm.gl module:"  << std::endl;

    _initialized = false;

    scm::out() << scm::log_level(scm::logging::ll_info)
               << "successfully shut down scm.gl module"  << std::endl;

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
                                  << gl_mod.context_information()._version_info << std::endl
               << "extensions : " << "(found " << gl_mod._supported_extensions.size() << ")" << std::endl;

    for (gl_module::extension_string_container::const_iterator i = gl_mod._supported_extensions.begin(); i != gl_mod._supported_extensions.end(); ++i) {
        out_stream << "             " << *i << std::endl;
    }

    return (out_stream);
}

} // namespace gl
} // namespace scm
