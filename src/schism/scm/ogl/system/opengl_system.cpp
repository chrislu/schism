
#include "opengl_system.h"

#include <cassert>

#include <scm/core.h>
#include <scm/ogl/gl.h>

namespace {
} // namespace

using namespace scm::gl;

opengl_system::opengl_system()
{
}

opengl_system::~opengl_system()
{
}

bool opengl_system::initialize()
{
    if (_initialized) {
        console.get() << con::log_level(con::warning)
                      << "opengl_system::initialize(): "
                      << "allready initialized" << std::endl;
        return (true);
    }

    if (glewInit() != GLEW_OK) {
        console.get() << con::log_level(con::error)
                      << "opengl_system::initialize: "
                      << "unable to initialize OpenGL sub system (GLEW Init failed)" << std::endl;

        return (false);
    }


    _initialized = true;
    return (true);
}

bool opengl_system::shutdown()
{

    _initialized = false;
    return (true);
}

bool opengl_system::is_supported(const std::string& ext) const
{
    return (glewIsSupported(ext.c_str()) == GL_TRUE ? true : false);
}
