
#ifndef SCM_OGL_VERTEXBUFFER_INCLUDED
#define SCM_OGL_VERTEXBUFFER_INCLUDED

#include <cstddef>
#include <string>

#include <scm/ogl/gl.h>

#include <scm/ogl/vertexbuffer_object/vertex_array.h>
#include <scm/ogl/vertexbuffer_object/vertex_layout.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class __scm_export(ogl) vertexbuffer
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

    const gl::vertex_layout&    get_vertex_layout() const;

protected:
    vertex_array                _vertices;

    std::size_t                 _num_vertices;

    vertex_layout               _vertex_layout;

private:

}; // class vertexbuffer

} // namespace gl
} // namespace scm

#endif // SCM_OGL_VERTEXBUFFER_H_INCLUDED
