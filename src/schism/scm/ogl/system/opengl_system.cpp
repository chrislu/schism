
#include "opengl_system.h"

#include <cassert>

#include <scm/core.h>

namespace {
static std::string      font_manager_name = std::string("gl_font_manager");
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

    _font_manager.reset(new font_resource_manager());

    root.get().register_resource_manager(font_manager_name, _font_manager.get());

    _initialized = true;
    return (true);
}

bool opengl_system::shutdown()
{
    root.get().unregister_resource_manager(font_manager_name);

    _initialized = false;
    return (true);
}

bool opengl_system::is_supported(const std::string& ext) const
{
    return (glewIsSupported(ext.c_str()) == GL_TRUE ? true : false);
}

font_resource_manager& opengl_system::get_font_manager()
{
    assert(_font_manager.get() != 0);

    return (*_font_manager.get());
}
