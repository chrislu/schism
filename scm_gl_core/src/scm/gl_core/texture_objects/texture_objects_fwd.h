
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_TEXTURE_OBJECTS_FWD_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_OBJECTS_FWD_H_INCLUDED

#include <scm/core/memory.h>

namespace scm {
namespace gl {

class texture;

class texture_image;

class texture_handle;

struct texture_1d_desc;
class  texture_1d;

struct texture_2d_desc;
class  texture_2d;

struct texture_3d_desc;
class  texture_3d;

struct texture_cube_desc;
class  texture_cube;

struct texture_buffer_desc;
class  texture_buffer;

typedef shared_ptr<texture>                 texture_ptr;
typedef shared_ptr<texture const>           texture_cptr;
typedef shared_ptr<texture_image>           texture_image_ptr;
typedef shared_ptr<texture_image const>     texture_image_cptr;
typedef shared_ptr<texture_handle>          texture_handle_ptr;
typedef shared_ptr<texture_handle const>    texture_handle_cptr;
typedef shared_ptr<texture_1d>              texture_1d_ptr;
typedef shared_ptr<texture_1d const>        texture_1d_cptr;
typedef shared_ptr<texture_2d>              texture_2d_ptr;
typedef shared_ptr<texture_2d const>        texture_2d_cptr;
typedef shared_ptr<texture_3d>              texture_3d_ptr;
typedef shared_ptr<texture_3d const>        texture_3d_cptr;
typedef shared_ptr<texture_cube>            texture_cube_ptr;
typedef shared_ptr<texture_cube const>      texture_cube_cptr;
typedef shared_ptr<texture_buffer>          texture_buffer_ptr;
typedef shared_ptr<texture_buffer const>    texture_buffer_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_TEXTURE_OBJECTS_FWD_H_INCLUDED
