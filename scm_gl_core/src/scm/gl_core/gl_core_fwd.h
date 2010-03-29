
#ifndef SCM_GL_CORE_GL_CORE_FWD_H_INCLUDED
#define SCM_GL_CORE_GL_CORE_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class render_device;
class render_context;
class render_device_child;
class render_device_resource;

typedef shared_ptr<render_device>   render_device_ptr;
typedef shared_ptr<render_context>  render_context_ptr;

class buffer;
class shader;
class program;

typedef shared_ptr<buffer>  buffer_ptr;
typedef shared_ptr<shader>  shader_ptr;
typedef shared_ptr<program> program_ptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_GL_CORE_FWD_H_INCLUDED
