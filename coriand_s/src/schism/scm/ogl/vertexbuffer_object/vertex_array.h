
#ifndef SCM_OGL_VERTEX_ARRAY_H_INCLUDED
#define SCM_OGL_VERTEX_ARRAY_H_INCLUDED

#include <scm/ogl/buffer_object/buffer_object.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class __scm_export(ogl) vertex_array : public buffer_object
{
public:
    vertex_array();
    virtual ~vertex_array();

}; // class element_array

} // namespace gl
} // namespace scm

#endif // SCM_OGL_VERTEX_ARRAY_H_INCLUDED
