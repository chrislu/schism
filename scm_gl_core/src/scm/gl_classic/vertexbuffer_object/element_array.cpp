
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "element_array.h"

#include <scm/gl_classic/opengl.h>

namespace scm {
namespace gl_classic {

element_array::element_array()
: buffer_object(GL_ELEMENT_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER_BINDING)
{
}

element_array::~element_array()
{
}

} // namespace gl_classic
} // namespace scm
