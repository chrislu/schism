
#include "render_context.h"

#include <scm/gl_core/shader.h>
#include <scm/gl_core/program.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/object_state.h>
#include <scm/gl_core/opengl/gl3_core.h>

namespace scm {
namespace gl {

render_context::render_context(render_device& dev)
  : render_device_child(dev),
    _opengl_api_core(dev.opengl3_api())
{
}

render_context::~render_context()
{
}

const opengl::gl3_core&
render_context::opengl_api() const
{
    return (_opengl_api_core);
}

void
render_context::apply()
{
    apply_program();
}

// buffer api /////////////////////////////////////////////////////////////////////////////////

// shader api /////////////////////////////////////////////////////////////////////////////////
void
render_context::bind_program(const program_ptr& p)
{
    _current_state._program = p;
}

program_ptr
render_context::current_program() const
{
    return (_current_state._program);
}

void
render_context::apply_program()
{
    if (!_current_state._program) {
        state().set(object_state::OS_ERROR_INVALID_VALUE);
        return;
    }
    if (!_current_state._program->ok()) {
        state().set(object_state::OS_ERROR_INVALID_OPERATION);
        return;
    }

    // bind the program
    _current_state._program->bind(*this);
    _applied_state._program = _current_state._program;

    // bind uniforms
    _applied_state._program->bind_uniforms(*this);
}

} // namespace gl
} // namespace scm
