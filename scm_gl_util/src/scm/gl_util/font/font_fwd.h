
#ifndef SCM_GL_UTIL_FONT_FWD_H_INCLUDED
#define SCM_GL_UTIL_FONT_FWD_H_INCLUDED

#include <scm/core/pointer_types.h>

namespace scm {
namespace gl {

class font_face;
class font_renderer;

typedef shared_ptr<font_face>           font_face_ptr;
typedef shared_ptr<const font_face>     font_face_cptr;

typedef shared_ptr<font_renderer>       font_renderer_ptr;
typedef shared_ptr<const font_renderer> font_renderer_cptr;

} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_FONT_FWD_H_INCLUDED
