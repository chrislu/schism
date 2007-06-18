
#include "volume_renderer_crossplanes_glsl.h"

#include <scm/core/math/math.h>
#include <scm/core/math/math_gl.h>

#include <scm/ogl/gl.h>

#include <volume_renderer/volume_renderer_parameters.h>

#include <iostream> // debug

namespace gl
{
    volume_renderer_crossplanes_glsl::volume_renderer_crossplanes_glsl()
        : _program_obj(),
          _vert_shader_obj(GL_VERTEX_SHADER),
          _frag_shader_obj(GL_FRAGMENT_SHADER)
    {
    }

    volume_renderer_crossplanes_glsl::~volume_renderer_crossplanes_glsl()
    {
        this->shutdown();
    }

    bool volume_renderer_crossplanes_glsl::initialize()
    {
        static bool initialized = false;

        if (initialized) {
            return (true);
        }

        if (!volume_renderer::initialize()) {
            return (false);
        }

        bool compile_error = false;
        bool attach_error  = false;
        bool link_error    = false;


        std::cout << "volume_renderer_crossplanes_glsl:" << std::endl;
        std::cout << "starting compile shader variants:" << std::endl;
        // back face shaders
        if (!_vert_shader_obj.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_cp_vert.glsl")) {
            std::cout << "Error opening vertex shader source file: ";
            std::cout << "./volume_renderer/shader/vol_crossplanes_vert.glsl" << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_cp_frag.glsl")) {
            std::cout << "Error opening fragment shader source file: ";
            std::cout << "./volume_renderer/shader/vol_crossplanes_frag.glsl" << std::endl;
            compile_error = true;
        }

        //if (!_frag_shader_obj.add_include_code_from_file("./volume_renderer/shader/vol_determine_uncertainty.glsl")) {
        //    std::cout << "Error opening fragment shader source file: ";
        //    std::cout << "./volume_renderer/shader/vol_determine_uncertainty.glsl" << std::endl;
        //    compile_error = true;
        //}

        if (!_vert_shader_obj.compile()) {
            std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
            std::cout << _vert_shader_obj.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj.compile()) {
            std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
            std::cout << _frag_shader_obj.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (compile_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj.attach_shader(_vert_shader_obj)) {
            std::cout << "unable to attach vertex shader to program object:" << std::endl;
            attach_error = true;
        }

        if (!_program_obj.attach_shader(_frag_shader_obj)) {
            std::cout << "unable to attach fragment shader to program object:" << std::endl;
            attach_error = true;
        }

        if (attach_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj.link()) {
            std::cout << "Error linking program - linker output:" << std::endl;
            std::cout << _program_obj.get_linker_output() << std::endl;
            link_error = true;
        }

        if (link_error) {
            shutdown();
            return (false);
        }
        return (true);
    }

    void volume_renderer_crossplanes_glsl::frame(const gl::volume_renderer_parameters& params)
    {

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        glActiveTexture(GL_TEXTURE0);
        params._volume_texture.bind();
        glActiveTexture(GL_TEXTURE1);
        params._color_alpha_texture.bind();
        //glActiveTexture(GL_TEXTURE2);
        //params._uncertainty_texture.bind();

        //glBindTexture(GL_TEXTURE_3D, get_parameters().get_volume_texture().get_texture_id());
        //glEnable(GL_TEXTURE_3D);

        _planes.set_slice_x(params._cp_pos.x);
        _planes.set_slice_y(params._cp_pos.y);
        _planes.set_slice_z(params._cp_pos.z);

        static float light_dir[3] = { 1.f, 1.f, 1.f };
        static float light_dif[3] = { .26f, .3f, .4f };
        static float light_spc[3] = { 0.6f, 0.7f, .75f };


        _program_obj.bind();

        _program_obj.set_uniform_1i("_volume", 0);
        _program_obj.set_uniform_1i("_color_alpha", 1);
        //_program_obj.set_uniform_1i("_uncertainty", 2);
        //_program_obj.set_uniform_1i("_max_loop_count", 255);
        //_program_obj.set_uniform_3f("up_direction", 0, 0, 1);

        //_program_obj.set_uniform_3fv("_light_dir", 1, light_dir);
        //_program_obj.set_uniform_3fv("_light_col_diff", 1, light_dif);
        //_program_obj.set_uniform_3fv("_light_col_spec", 1, light_spc);

        glPushMatrix();
        {
            glScalef(params._aspect.x,
                     params._aspect.y,
                     params._aspect.z);
            glTranslatef(.5f, .5f, .5f);

            _planes.render(); 
        }
        glPopMatrix();

        _program_obj.unbind();

        //glDisable(GL_TEXTURE_3D);
        //glBindTexture(GL_TEXTURE_3D, 0);
        //params._uncertainty_texture.unbind();
        glActiveTexture(GL_TEXTURE1);
        params._color_alpha_texture.unbind();
        glActiveTexture(GL_TEXTURE0);
        params._volume_texture.unbind();

        glDisable(GL_BLEND);

    }

    bool volume_renderer_crossplanes_glsl::shutdown()
    {
        return (true);
    }

} // namespace gl
