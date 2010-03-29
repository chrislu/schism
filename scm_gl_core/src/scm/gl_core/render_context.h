
#ifndef SCM_GL_CORE_RENDER_CONTEXT_H_INCLUDED
#define SCM_GL_CORE_RENDER_CONTEXT_H_INCLUDED

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/render_device_child.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

namespace opengl {
class gl3_core;
} // namespace opengl

class render_device;

class __scm_export(gl_core) render_context : public render_device_child
{
////// types //////////////////////////////////////////////////////////////////////////////////////
private:
    struct binding_state_type {
        program_ptr                 _program;

        // _textures[];
        // _vertex_attribute_buffers[];
        // _index_buffer;
        // _framebuffer;
        // _queries;
    }; // struct binding_state_type

////// methods ////////////////////////////////////////////////////////////////////////////////////
public:
    virtual ~render_context();

    const opengl::gl3_core&     opengl_api() const;
    void                        apply();

    // buffer api /////////////////////////////////////////////////////////////////////////////////
public:

    // shader api /////////////////////////////////////////////////////////////////////////////////
public:
    void                        bind_program(const program_ptr& p);
    program_ptr                 current_program() const;

protected:
    void                        apply_program();

protected:

    // texture api ////////////////////////////////////////////////////////////////////////////////
public:
    //

protected:
    render_context(render_device& dev);

private:
    const opengl::gl3_core&     _opengl_api_core;

    binding_state_type          _current_state;
    binding_state_type          _applied_state;

    // TODO
    //program_ptr                 _default_program;

    friend class render_device;    
}; // class render_context

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_RENDER_CONTEXT_H_INCLUDED
