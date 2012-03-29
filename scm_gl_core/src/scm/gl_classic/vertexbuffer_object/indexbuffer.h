
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_OGL_INDEXBUFFER_H_INCLUDED
#define SCM_OGL_INDEXBUFFER_H_INCLUDED

#include <cstddef>
#include <string>

#include <scm/gl_classic/opengl.h>

#include <scm/gl_classic/vertexbuffer_object/element_array.h>
#include <scm/gl_classic/vertexbuffer_object/element_layout.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) indexbuffer
{
public:
    indexbuffer();
    virtual ~indexbuffer();

    void                        bind() const;
    void                        draw_elements() const;
    void                        unbind() const;
                            
    void                        clear();
                            
    bool                        element_data(std::size_t            /*num_indices*/,
                                             const element_layout&  /*layout*/,
                                             const void*const       /*data*/,
                                             unsigned               /*usage_type*/ = GL_STATIC_DRAW);

    std::size_t                 num_indices() const;

    const gl_classic::element_layout&   get_element_layout() const;

protected:
    element_array               _indices;

    std::size_t                 _num_indices;

    element_layout              _element_layout;

private:

}; // class indexbuffer

} // namespace gl_classic
} // namespace scm

#endif // SCM_OGL_INDEXBUFFER_H_INCLUDED
