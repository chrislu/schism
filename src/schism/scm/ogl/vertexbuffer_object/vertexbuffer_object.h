
#ifndef SCM_OGL_VERTEXBUFFER_OBJECT_H_INCLUDED
#define SCM_OGL_VERTEXBUFFER_OBJECT_H_INCLUDED

#include <cstddef>
#include <string>

#include <scm/ogl/gl.h>

#include <scm/ogl/vertexbuffer_object/element_array.h>
#include <scm/ogl/vertexbuffer_object/element_layout.h>
#include <scm/ogl/vertexbuffer_object/vertex_array.h>
#include <scm/ogl/vertexbuffer_object/vertex_layout.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class __scm_export(ogl) vertexbuffer_object
{
public:
    vertexbuffer_object();
    virtual ~vertexbuffer_object();

    void                        bind() const;
    void                        draw_elements() const;
    void                        unbind() const;
                            
    void                        clear();
                            
    bool                        vertex_data(std::size_t            /*num_vertices*/,
                                            const vertex_layout&   /*layout*/,
                                            const void*const       /*data*/,
                                            unsigned               /*usage_type*/ = GL_STATIC_DRAW);
    bool                        element_data(std::size_t            /*num_indices*/,
                                             const element_layout&  /*layout*/,
                                             const void*const       /*data*/,
                                             unsigned               /*usage_type*/ = GL_STATIC_DRAW);
                            
    std::size_t                 num_vertices() const;
    std::size_t                 num_indices() const;

    const gl::vertex_layout&    get_vertex_layout() const;
    const gl::element_layout&   get_element_layout() const;

protected:
    vertex_array                _vertices;
    element_array               _indices;

    std::size_t                 _num_vertices;
    std::size_t                 _num_indices;

    vertex_layout               _vertex_layout;
    element_layout              _element_layout;

private:

}; // class vertexbuffer_object

} // namespace gl
} // namespace scm

#endif // SCM_OGL_VERTEXBUFFER_OBJECT_H_INCLUDED
