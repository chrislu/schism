
#ifndef VOLUME_RENDERER_RAYCAST_GLSL_H_INCLUDED
#define VOLUME_RENDERER_RAYCAST_GLSL_H_INCLUDED

// includes, system

// includes, project
#include <volume_renderer/volume_renderer.h>

#include <ogl/shader_objects/program_object.h>
#include <ogl/shader_objects/shader_object.h>

#include <ogl/utilities/volume_textured_unit_cube.h>

namespace gl
{
    class volume_renderer_raycast_glsl : public volume_renderer
    {
    public:
        volume_renderer_raycast_glsl();
        virtual ~volume_renderer_raycast_glsl();

        bool                    initialize();
        void                    frame(const gl::volume_renderer_parameters&);

        void                    do_inside_pass(bool f) {_do_inside_pass = f;}

    protected:
        bool                    shutdown();

    private:
        gl::program_object      _program_obj_front_pass;
        gl::shader_object       _vert_shader_obj_front_pass;
        gl::shader_object       _frag_shader_obj_front_pass;

        gl::program_object      _program_obj_back_pass;
        gl::shader_object       _vert_shader_obj_back_pass;
        gl::shader_object       _frag_shader_obj_back_pass;

        gl::program_object      _program_obj_cp;
        gl::shader_object       _vert_shader_obj_cp;
        gl::shader_object       _frag_shader_obj_cp;

        bool                    _do_inside_pass;

    }; // class volume_renderer

} // namespace gl

#endif // VOLUME_RENDERER_RAYCAST_GLSL_H_INCLUDED
