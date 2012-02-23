
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_RENDER_DEVICE_FWD_H_INCLUDED
#define SCM_GL_CORE_RENDER_DEVICE_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace gl {

class render_device;
class render_context;
class render_device_child;
class render_device_resource;

typedef shared_ptr<render_device>           render_device_ptr;
typedef shared_ptr<const render_device>     render_device_cptr;
typedef weak_ptr<render_device>             render_device_wptr;
typedef shared_ptr<render_context>          render_context_ptr;
typedef shared_ptr<const render_context>    render_context_cptr;
typedef weak_ptr<render_context>            render_context_wptr;

class context_program_guard;
class context_vertex_input_guard;
class context_state_objects_guard;
class context_texture_units_guard;
class context_framebuffer_guard;
class context_all_guard;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_RENDER_DEVICE_FWD_H_INCLUDED
