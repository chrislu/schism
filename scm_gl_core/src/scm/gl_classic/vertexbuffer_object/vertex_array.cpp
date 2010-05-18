
#include "vertex_array.h"

#include <scm/gl_classic/opengl.h>

namespace scm {
namespace gl_classic {

vertex_array::vertex_array()
: buffer_object(GL_ARRAY_BUFFER, GL_ARRAY_BUFFER_BINDING)
{
}

vertex_array::~vertex_array()
{
}

} // namespace gl_classic
} // namespace scm
