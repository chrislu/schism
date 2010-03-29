
#include "render_device.h"

#include <exception>
#include <stdexcept>
#include <sstream>

#include <boost/bind.hpp>

#include <scm/gl_core/log.h>
#include <scm/gl_core/program.h>
#include <scm/gl_core/render_context.h>
#include <scm/gl_core/opengl/config.h>
#include <scm/gl_core/opengl/gl3_core.h>

namespace scm {
namespace gl {

render_device::render_device()
{
    _opengl3_api_core.reset(new opengl::gl3_core());

    if (!_opengl3_api_core->initialize()) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core.";
        glerr() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
    if (!(   _opengl3_api_core->context_information()._version_major >= 3
          && _opengl3_api_core->context_information()._version_minor >= 2)) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core "
          << "(at least OpenGL 3.2 requiered, encountered version "
          << _opengl3_api_core->context_information()._version_major << "."
          << _opengl3_api_core->context_information()._version_minor << ").";
        glerr() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
#ifdef SCM_GL_CORE_USE_DIRECT_STATE_ACCESS
    if (!_opengl3_api_core->is_supported("GL_EXT_direct_state_access")) {
        std::ostringstream s;
        s << "render_device::render_device(): error initializing gl core "
          << "(missing requiered extension GL_EXT_direct_state_access).";
        glerr() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }
#endif

    // setup main rendering context
    _main_context.reset(new render_context(*this));
}

render_device::~render_device()
{
}

const opengl::gl3_core&
render_device::opengl3_api() const
{
    return (*_opengl3_api_core);
}

render_context_ptr
render_device::main_context() const
{
    return (_main_context);
}

buffer_ptr
render_device::create_buffer(const buffer::descriptor_type&  buffer_desc,
                             const void*                     initial_data)
{
    buffer_ptr new_buffer(new buffer(*this, buffer_desc, initial_data),
                          boost::bind(&render_device::release_resource, this, _1));
    if (new_buffer->fail()) {
        if (new_buffer->bad()) {
            glerr() << log::error << "render_device::create_buffer(): unable to create buffer object ("
                    << new_buffer->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << log::error << "render_device::create_buffer(): unable to allocate buffer ("
                    << new_buffer->state().state_string() << ")." << log::end;
        }
        return (buffer_ptr());
    }
    else {
        register_resource(new_buffer.get());
        return (new_buffer);
    }
}

shader_ptr
render_device::create_shader(shader::stage_type t,
                             const std::string& s)
{
    shader_ptr new_shader(new shader(*this, t, s));
    if (new_shader->fail()) {
        if (new_shader->bad()) {
            glerr() << "render_device::create_shader(): unable to create shader object ("
                    << new_shader->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << "render_device::create_shader(): unable to compile shader ("
                    << new_shader->state().state_string() << "):" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return (shader_ptr());
    }
    else {
        if (!new_shader->info_log().empty()) {
            glout() << log::info << "render_device::create_shader(): compiler info" << log::nline
                    << new_shader->info_log() << log::end;
        }
        return (new_shader);
    }
}

program_ptr
render_device::create_program(const shader_list& in_shaders)
{
    program_ptr new_program(new program(*this, in_shaders));
    if (new_program->fail()) {
        if (new_program->bad()) {
            glerr() << "render_device::create_program(): unable to create shader object ("
                    << new_program->state().state_string() << ")." << log::end;
        }
        else {
            glerr() << "render_device::create_program(): error during link operation ("
                    << new_program->state().state_string() << "):" << log::nline
                    << new_program->info_log() << log::end;
        }
        return (program_ptr());
    }
    else {
        if (!new_program->info_log().empty()) {
            glout() << log::info << "render_device::create_program(): linker info" << log::nline
                    << new_program->info_log() << log::end;
        }
        return (new_program);
    }
}

void
render_device::print_device_informations(std::ostream& os) const
{
    os << "OpenGL render device" << std::endl;
    os << *_opengl3_api_core;
}

void
render_device::register_resource(render_device_resource* res_ptr)
{
    _registered_resources.insert(res_ptr);
}

void
render_device::release_resource(render_device_resource* res_ptr)
{
    resource_ptr_set::iterator res_iter = _registered_resources.find(res_ptr);
    if (res_iter != _registered_resources.end()) {
        _registered_resources.erase(res_iter);
    }

    delete res_ptr;
}

std::ostream& operator<<(std::ostream& os, const render_device& ren_dev)
{
    ren_dev.print_device_informations(os);
    return (os);
}

} // namespace gl
} // namespace scm
