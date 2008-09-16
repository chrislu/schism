
#include "gl.h"

#include <boost/scoped_ptr.hpp>

#include <scm/core.h>
#include <scm/ogl.h>
#include <scm/ogl/system/opengl_system.h>

namespace {

static bool scm_gl_initialized = false;

static scm::scoped_ptr<scm::gl::opengl_system>  scm_ogl_sys;

static std::string  scm_ogl_sys_name = std::string("ogl");

} // namespace


namespace scm {

core::core_system_singleton<gl::opengl_system>::type  ogl = core::core_system_singleton<gl::opengl_system>::type();

namespace gl {

bool initialize()
{
    if (scm_gl_initialized) {
        console.get() << con::log_level(con::warning)
                      << "scm::gl::initialize(): "
                      << "allready initialized" << std::endl;
        return (true);
    }

    console.get() << con::log_level(con::info)
                  << "initializing scm::ogl library:"  << std::endl;

    scm_ogl_sys.reset(new scm::gl::opengl_system());

    console.get() << con::log_level(con::info)
                  << " - initializing: '" << scm_ogl_sys_name << "'" << std::endl;

    if (!scm_ogl_sys->initialize()) {
        scm_ogl_sys.reset();
        console.get() << con::log_level(con::error)
                      << "   - scm::gl::initialize(): "
                      << "subsystem '" << scm_ogl_sys_name << "' initialize returned with error" << std::endl;

        return (false);
    }

    root.get().register_subsystem(scm_ogl_sys_name, scm_ogl_sys.get());

    ogl.set_instance(scm_ogl_sys.get());

    console.get() << con::log_level(con::info)
                  << "successfully initialized scm::ogl library"  << std::endl;
    scm_gl_initialized = true;
    return (true);
}

bool shutdown()
{
    if (!scm_gl_initialized) {
        return (true);
    }

    console.get() << con::log_level(con::info)
                  << "shutting down scm::ogl library:"  << std::endl;

    assert(scm_ogl_sys.get() != 0);

    root.get().unregister_subsystem(scm_ogl_sys_name);

    console.get() << con::log_level(con::info)
                  << " - shutting down: '" << scm_ogl_sys_name << "'" << std::endl;
    bool ret = scm_ogl_sys->shutdown();
    if (!ret) {
        console.get() << con::log_level(con::error)
                      << "   - scm::gl:shutdown(): "
                      << "subsystem '" << scm_ogl_sys_name << "' shutdown returned with error" << std::endl;
    }

    scm_ogl_sys.reset();

    console.get() << con::log_level(con::info)
                  << "successfully shut down scm::ogl library"  << std::endl;

    scm_gl_initialized = false;
    return (ret);
}


} // namespace gl
} // namespace scm
