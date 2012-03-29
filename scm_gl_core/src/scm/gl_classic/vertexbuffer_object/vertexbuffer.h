
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_OGL_VERTEXBUFFER_INCLUDED
#define SCM_OGL_VERTEXBUFFER_INCLUDED

#include <cstddef>
#include <string>

#include <scm/gl_classic/opengl.h>

#include <scm/gl_classic/vertexbuffer_object/vertex_array.h>
#include <scm/gl_classic/vertexbuffer_object/vertex_layout.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) vertexbuffer
{
public:
    vertexbuffer();
    virtual ~vertexbuffer();

    void                        bind() const;
    void                        unbind() const;
                            
    void                        clear();
                            
    bool                        vertex_data(std::size_t            /*num_vertices*/,
                                            const vertex_layout&   /*layout*/,
                                            const void*const       /*data*/,
                                            unsigned               /*usage_type*/ = GL_STATIC_DRAW);

    std::size_t                 num_vertices() const;

    const gl_classic::vertex_layout&    get_vertex_layout() const;

protected:
    vertex_array                _vertices;

    std::size_t                 _num_vertices;

    vertex_layout               _vertex_layout;

private:

}; // class vertexbuffer

} // namespace gl_classic
} // namespace scm

#endif // SCM_OGL_VERTEXBUFFER_H_INCLUDED
