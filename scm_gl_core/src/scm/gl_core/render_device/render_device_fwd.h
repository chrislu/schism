
#ifndef SCM_GL_CORE_RENDER_DEVICE_FWD_H_INCLUDED
#define SCM_GL_CORE_RENDER_DEVICE_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class render_device;
class render_context;
class render_device_child;
class render_device_resource;

typedef shared_ptr<render_device>   render_device_ptr;
typedef weak_ptr<render_device>     render_device_weak_ptr;
typedef shared_ptr<render_context>  render_context_ptr;
typedef weak_ptr<render_context>    render_context_weak_ptr;

class context_program_guard;
class context_vertex_input_guard;
class context_state_objects_guard;
class context_texture_units_guard;
class context_framebuffer_guard;
class context_all_guard;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_RENDER_DEVICE_FWD_H_INCLUDED
