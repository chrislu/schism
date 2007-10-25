
#include "vertex_array.h"

#include <scm/ogl/gl.h>

namespace scm {
namespace gl {

vertex_array::vertex_array()
: buffer_object(GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING)
{
}

vertex_array::~vertex_array()
{
}

} // namespace gl
} // namespace scm
