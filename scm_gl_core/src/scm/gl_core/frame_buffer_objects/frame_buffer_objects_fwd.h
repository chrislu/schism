
#ifndef SCM_GL_CORE_FRAME_BUFFER_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_FRAME_BUFFER_OBJECTS_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class render_target;

struct render_buffer_desc;
class  render_buffer;

class frame_buffer;

class viewport;

typedef shared_ptr<render_target>   render_target_ptr;
typedef shared_ptr<render_buffer>   render_buffer_ptr;
typedef shared_ptr<frame_buffer>    frame_buffer_ptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_FRAME_BUFFER_OBJECTS_FWD_H_INCLUDED
