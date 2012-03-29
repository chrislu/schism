
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_OGL_VERTEX_ARRAY_H_INCLUDED
#define SCM_OGL_VERTEX_ARRAY_H_INCLUDED

#include <scm/gl_classic/buffer_object/buffer_object.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) vertex_array : public buffer_object
{
public:
    vertex_array();
    virtual ~vertex_array();

}; // class element_array

} // namespace gl_classic
} // namespace scm

#endif // SCM_OGL_VERTEX_ARRAY_H_INCLUDED
