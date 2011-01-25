
#ifndef SCM_GL_UTIL_VIEWER_FWD_H_INCLUDED
#define SCM_GL_UTIL_VIEWER_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class camera;
class camera_uniform_block;
class viewer;

typedef shared_ptr<camera_uniform_block>        camera_uniform_block_ptr;
typedef shared_ptr<camera_uniform_block const>  camera_uniform_block_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_VIEWER_FWD_H_INCLUDED
