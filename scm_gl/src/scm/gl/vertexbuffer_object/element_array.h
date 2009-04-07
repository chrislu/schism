
#ifndef SCM_OGL_ELEMENT_ARRAY_H_INCLUDED
#define SCM_OGL_ELEMENT_ARRAY_H_INCLUDED

#include <scm/gl/buffer_object/buffer_object.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class __scm_export(ogl) element_array : public buffer_object
{
public:
    element_array();
    virtual ~element_array();

}; // class element_array

} // namespace gl
} // namespace scm

#endif // SCM_OGL_ELEMENT_ARRAY_H_INCLUDED
