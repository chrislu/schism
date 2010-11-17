
#ifndef SCM_GL_CORE_TEXTURE_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_OBJECTS_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class texture;

class texture_image;

struct texture_1d_desc;
class  texture_1d;

struct texture_2d_desc;
class  texture_2d;

struct texture_3d_desc;
class  texture_3d;

struct texture_buffer_desc;
class  texture_buffer;

typedef shared_ptr<texture>                 texture_ptr;
typedef shared_ptr<texture const>           texture_cptr;
typedef shared_ptr<texture_image>           texture_image_ptr;
typedef shared_ptr<texture_image const>     texture_image_cptr;
typedef shared_ptr<texture_1d>              texture_1d_ptr;
typedef shared_ptr<texture_1d const>        texture_1d_cptr;
typedef shared_ptr<texture_2d>              texture_2d_ptr;
typedef shared_ptr<texture_2d const>        texture_2d_cptr;
typedef shared_ptr<texture_3d>              texture_3d_ptr;
typedef shared_ptr<texture_3d const>        texture_3d_cptr;
typedef shared_ptr<texture_buffer>          texture_buffer_ptr;
typedef shared_ptr<texture_buffer const>    texture_buffer_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_TEXTURE_OBJECTS_FWD_H_INCLUDED
