
#include "element_array.h"

#include <scm/gl/opengl.h>

namespace scm {
namespace gl {

element_array::element_array()
: buffer_object(GL_ELEMENT_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER_BINDING)
{
}

element_array::~element_array()
{
}

} // namespace gl
} // namespace scm
