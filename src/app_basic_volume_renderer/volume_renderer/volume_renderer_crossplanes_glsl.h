
#ifndef VOLUME_RENDERER_CROSSPLANES_GLSL_H_INCLUDED
#define VOLUME_RENDERER_CROSSPLANES_GLSL_H_INCLUDED

// includes, system

// includes, project
#include <volume_renderer/volume_renderer.h>

#include <scm/ogl/shader_objects/program_object.h>
#include <scm/ogl/shader_objects/shader_object.h>

namespace gl
{
    class volume_renderer_crossplanes_glsl : public volume_renderer
    {
    public:
        volume_renderer_crossplanes_glsl();
        virtual ~volume_renderer_crossplanes_glsl();

        bool                    initialize();
        void                    frame(const gl::volume_renderer_parameters&);

    protected:
        bool                    shutdown();

    private:
        scm::gl::program_object      _program_obj;
        scm::gl::shader_object       _vert_shader_obj;
        scm::gl::shader_object       _frag_shader_obj;

    }; // class volume_renderer_crossplanes_glsl

} // namespace gl

#endif // VOLUME_RENDERER_CROSSPLANES_GLSL_H_INCLUDED
