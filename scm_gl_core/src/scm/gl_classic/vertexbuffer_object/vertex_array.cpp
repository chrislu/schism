
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

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
