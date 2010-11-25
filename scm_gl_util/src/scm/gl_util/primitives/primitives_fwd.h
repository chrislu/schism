
#ifndef SCM_GL_UTIL_PRIMITIVES_FWD_H_INCLUDED
#define SCM_GL_UTIL_PRIMITIVES_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class coordinate_cross;

typedef shared_ptr<coordinate_cross>                coordinate_cross_ptr;
typedef shared_ptr<const coordinate_cross>          coordinate_cross_cptr;

class geometry;
class geometry_highlight;
class box_geometry;
class quad_geometry;
class wavefront_obj_geometry;

typedef shared_ptr<geometry>                        geometry_ptr;
typedef shared_ptr<const geometry>                  geometry_cptr;
typedef shared_ptr<geometry_highlight>              geometry_highlight_ptr;
typedef shared_ptr<const geometry_highlight>        geometry_highlight_cptr;
typedef shared_ptr<box_geometry>                    box_geometry_ptr;
typedef shared_ptr<const box_geometry>              box_geometry_cptr;
typedef shared_ptr<quad_geometry>                   quad_geometry_ptr;
typedef shared_ptr<const quad_geometry>             quad_geometry_cptr;
typedef shared_ptr<wavefront_obj_geometry>          wavefront_obj_geometry_ptr;
typedef shared_ptr<const wavefront_obj_geometry>    wavefront_obj_geometry_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_PRIMITIVES_FWD_H_INCLUDED
