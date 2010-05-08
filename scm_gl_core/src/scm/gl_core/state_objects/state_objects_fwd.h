
#ifndef SCM_GL_CORE_STATE_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_STATE_OBJECTS_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

struct stencil_ops;
struct depth_stencil_state_desc;
class  depth_stencil_state;

class blend_state;

struct rasterizer_state_desc;
class  rasterizer_state;

struct sampler_state_desc;
class sampler_state;

typedef shared_ptr<depth_stencil_state> depth_stencil_state_ptr;
typedef shared_ptr<blend_state>         blend_state_ptr;
typedef shared_ptr<rasterizer_state>    rasterizer_state_ptr;
typedef shared_ptr<sampler_state>       sampler_state_ptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_STATE_OBJECTS_FWD_H_INCLUDED
