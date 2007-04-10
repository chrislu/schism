
#include "volume_renderer_raycast_glsl.h"

#include <scm_core/math/math.h>
#include <scm_core/math/math_gl.h>

#include <ogl/gl.h>

#include <iostream> // debug

#include <volume_renderer/volume_renderer_parameters.h>

namespace gl
{
    volume_renderer_raycast_glsl::volume_renderer_raycast_glsl()
        : _program_obj_front_pass(),
          _vert_shader_obj_front_pass(GL_VERTEX_SHADER),
          _frag_shader_obj_front_pass(GL_FRAGMENT_SHADER),
          _program_obj_back_pass(),
          _vert_shader_obj_back_pass(GL_VERTEX_SHADER),
          _frag_shader_obj_back_pass(GL_FRAGMENT_SHADER),
          _vert_shader_obj_cp(GL_VERTEX_SHADER),
          _frag_shader_obj_cp(GL_FRAGMENT_SHADER),
          _do_inside_pass(true)
    {
    }

    volume_renderer_raycast_glsl::~volume_renderer_raycast_glsl()
    {
        this->shutdown();
    }

    bool volume_renderer_raycast_glsl::initialize()
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

        std::cout << "starting compile of front face shader variants:" << std::endl;
        // front face shaders
        if (!_vert_shader_obj_front_pass.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_vert.glsl")) {
            std::cout << "Error opening vertex shader source file: ";
            std::cout << "./volume_renderer/shader/vol_raycast_vert.glsl" << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj_front_pass.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_frag.glsl")) {
            std::cout << "Error opening fragment shader source file: ";
            std::cout << "./volume_renderer/shader/vol_raycast_frag.glsl" << std::endl;
            compile_error = true;
        }

        //if (!_frag_shader_obj_front_pass.add_include_code_from_file("./volume_renderer/shader/vol_determine_uncertainty.glsl")) {
        //    std::cout << "Error opening fragment shader source file: ";
        //    std::cout << "./volume_renderer/shader/vol_determine_uncertainty.glsl" << std::endl;
        //    compile_error = true;
        //}

        _vert_shader_obj_front_pass.add_defines("#define RAYCAST_CORRECT_SAMPLE_START");
        _frag_shader_obj_front_pass.add_defines("#define RAYCAST_CORRECT_SAMPLE_START");

        if (!_vert_shader_obj_front_pass.compile()) {
            std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
            std::cout << _vert_shader_obj_front_pass.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj_front_pass.compile()) {
            std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
            std::cout << _frag_shader_obj_front_pass.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (compile_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj_front_pass.attach_shader(_vert_shader_obj_front_pass)) {
            std::cout << "unable to attach vertex shader to program object:" << std::endl;
            attach_error = true;
        }

        if (!_program_obj_front_pass.attach_shader(_frag_shader_obj_front_pass)) {
            std::cout << "unable to attach fragment shader to program object:" << std::endl;
            attach_error = true;
        }

        if (attach_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj_front_pass.link()) {
            std::cout << "Error linking program - linker output:" << std::endl;
            std::cout << _program_obj_front_pass.get_linker_output() << std::endl;
            link_error = true;
        }

        if (link_error) {
            shutdown();
            return (false);
        }

        std::cout << "starting compile of back face shader variants:" << std::endl;
        // back face shaders
        if (!_vert_shader_obj_back_pass.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_vert.glsl")) {
            std::cout << "Error opening vertex shader source file: ";
            std::cout << "./volume_renderer/shader/vol_raycast_vert.glsl" << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj_back_pass.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_frag.glsl")) {
            std::cout << "Error opening fragment shader source file: ";
            std::cout << "./volume_renderer/shader/vol_raycast_frag.glsl" << std::endl;
            compile_error = true;
        }

        //if (!_frag_shader_obj_back_pass.add_include_code_from_file("./volume_renderer/shader/vol_determine_uncertainty.glsl")) {
        //    std::cout << "Error opening fragment shader source file: ";
        //    std::cout << "./volume_renderer/shader/vol_determine_uncertainty.glsl" << std::endl;
        //    compile_error = true;
        //}

        _vert_shader_obj_back_pass.add_defines("#define RAYCAST_INSIDE_VOLUME\n #define RAYCAST_CORRECT_SAMPLE_START");
        _frag_shader_obj_back_pass.add_defines("#define RAYCAST_INSIDE_VOLUME\n #define RAYCAST_CORRECT_SAMPLE_START");

        if (!_vert_shader_obj_back_pass.compile()) {
            std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
            std::cout << _vert_shader_obj_back_pass.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj_back_pass.compile()) {
            std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
            std::cout << _frag_shader_obj_back_pass.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (compile_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj_back_pass.attach_shader(_vert_shader_obj_back_pass)) {
            std::cout << "unable to attach vertex shader to program object:" << std::endl;
            attach_error = true;
        }

        if (!_program_obj_back_pass.attach_shader(_frag_shader_obj_back_pass)) {
            std::cout << "unable to attach fragment shader to program object:" << std::endl;
            attach_error = true;
        }

        if (attach_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj_back_pass.link()) {
            std::cout << "Error linking program - linker output:" << std::endl;
            std::cout << _program_obj_back_pass.get_linker_output() << std::endl;
            link_error = true;
        }

        if (link_error) {
            shutdown();
            return (false);
        }


        std::cout << "starting compile of cross plane shader variant:" << std::endl;
        // back face shaders
        if (!_vert_shader_obj_cp.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_cp_vert.glsl")) {
            std::cout << "Error opening vertex shader source file: ";
            std::cout << "./volume_renderer/shader/vol_raycast_cp_vert.glsl" << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj_cp.set_source_code_from_file("./../../../src/app_basic_volume_renderer/volume_renderer/shader/vol_raycast_cp_frag.glsl")) {
            std::cout << "Error opening fragment shader source file: ";
            std::cout << "./volume_renderer/shader/vol_raycast_cp_frag.glsl" << std::endl;
            compile_error = true;
        }

        if (!_vert_shader_obj_cp.compile()) {
            std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
            std::cout << _frag_shader_obj_cp.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (!_frag_shader_obj_cp.compile()) {
            std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
            std::cout << _frag_shader_obj_cp.get_compiler_output() << std::endl;
            compile_error = true;
        }

        if (compile_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj_cp.attach_shader(_vert_shader_obj_cp)) {
            std::cout << "unable to attach vertex shader to program object:" << std::endl;
            attach_error = true;
        }

        if (!_program_obj_cp.attach_shader(_frag_shader_obj_cp)) {
            std::cout << "unable to attach fragment shader to program object:" << std::endl;
            attach_error = true;
        }

        if (attach_error) {
            shutdown();
            return (false);
        }

        if (!_program_obj_cp.link()) {
            std::cout << "Error linking program - linker output:" << std::endl;
            std::cout << _program_obj_cp.get_linker_output() << std::endl;
            link_error = true;
        }

        if (link_error) {
            shutdown();
            return (false);
        }

        return (true);
    }

    void volume_renderer_raycast_glsl::frame(const gl::volume_renderer_parameters& params)
    {

#if 1
        _planes.set_slice_x(params._cross_plane_positions.x);
        _planes.set_slice_y(params._cross_plane_positions.y);
        _planes.set_slice_z(params._cross_plane_positions.z);

        math::mat_glf_t texture_transform = math::mat4x4f_identity;

        texture_transform.scale((math::vec3f_t(1.0f) / params._aspect));
        texture_transform.translate(params._point_of_interest + math::vec3f_t(-0.5f));
        texture_transform.scale(params._aspect);
        texture_transform.translate(math::vec3f_t(+0.5f));
        texture_transform.scale(params._extend);
        texture_transform.translate(math::vec3f_t(-0.5f));

        math::vec4f_t   tex_min_bound = math::clamp(texture_transform * math::vec4f_t(0, 0, 0, 1), math::vec4f_t(0, 0, 0, 1), math::vec4f_t(1, 1, 1, 1));
        math::vec4f_t   tex_max_bound = math::clamp(texture_transform * math::vec4f_t(1, 1, 1, 1), math::vec4f_t(0, 0, 0, 1), math::vec4f_t(1, 1, 1, 1));
        //math::vec4f_t   tex_min_bound = texture_transform * math::vec4f_t(0, 0, 0, 1);
        //math::vec4f_t   tex_max_bound = texture_transform * math::vec4f_t(1, 1, 1, 1);



        glMatrixMode(GL_TEXTURE);
        glPushMatrix();
        glLoadMatrixf(texture_transform.mat_array);
        glMatrixMode(GL_MODELVIEW);

        glActiveTexture(GL_TEXTURE0);
        params._volume_texture.bind();

        glActiveTexture(GL_TEXTURE1);
        params._color_alpha_texture.bind();

        glActiveTexture(GL_TEXTURE2);
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, params._geom_depth_texture_id);

        if (params._cp_enabled) {
            // cross planes
            _program_obj_cp.bind();

            _program_obj_cp.set_uniform_1i("_volume", 0);
            _program_obj_cp.set_uniform_1i("_color_alpha", 1);

            glPushMatrix();
                glScalef(params._aspect.x,
                         params._aspect.y,
                         params._aspect.z);
                //glTranslatef(-0.5f, -0.5f, -0.5f);
                glColor3f(0, 0, 0.1f);
                _planes.render(); 
            glPopMatrix();

            _program_obj_cp.unbind();
        }

        glPushMatrix();
        
        //glTranslatef();
        glTranslatef(params._point_of_interest.x - 0.5f,
                     params._point_of_interest.y - 0.5f,
                     params._point_of_interest.z - 0.5f);
        glScalef(params._extend.x,
                 params._extend.y,
                 params._extend.z);
        glScalef(params._aspect.x,
                 params._aspect.y,
                 params._aspect.z);
        glTranslatef(-0.5f, -0.5f, -0.5f);


#if 1
        math::vec3f_t   neutral_cross_plane_pos = math::vec3f_t(-1);
        math::vec4f_t   near_plane;
        math::vec4f_t   cam_pos_obj_space;
        math::mat_glf_t projection;
        math::mat_glf_t modelview;
        math::mat_glf_t tex_inv_transpose = math::transpose(math::inverse(texture_transform));
        math::mat_glf_t clip;

        math::get_gl_matrix(GL_PROJECTION_MATRIX, projection);
        math::get_gl_matrix(GL_MODELVIEW_MATRIX, modelview);

        clip = projection * modelview;

        // without ability to get the column vectors
        near_plane[0] = clip.m03 + clip.m02;
        near_plane[1] = clip.m07 + clip.m06;
        near_plane[2] = clip.m11 + clip.m10;
        near_plane[3] = clip.m15 + clip.m14;

        // with ability to get the column vectors
        //near_plane[0] = clip[0].w + clip[0].z;
        //near_plane[1] = clip[1].w + clip[1].z;
        //near_plane[2] = clip[2].w + clip[2].z;
        //near_plane[3] = clip[3].w + clip[3].z;

        float length = math::sqrt(   near_plane[0]*near_plane[0]
                                   + near_plane[1]*near_plane[1]
                                   + near_plane[2]*near_plane[2]);

        near_plane /= length;

        near_plane = tex_inv_transpose * near_plane;

        //glDisable(GL_MULTISAMPLE);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        //front pass
        _program_obj_front_pass.bind();

        _program_obj_front_pass.set_uniform_1i("_volume", 0);
        _program_obj_front_pass.set_uniform_1i("_color_alpha", 1);
        _program_obj_front_pass.set_uniform_1i("_geom_depth", 2);

        _program_obj_front_pass.set_uniform_1f("_step_size", 1.0f/params._step_size);
        _program_obj_front_pass.set_uniform_1i("_max_loop_counter", 255);

        _program_obj_front_pass.set_uniform_2fv("_screen_dimensions", 1, params._screen_dimensions.vec_array);

        _program_obj_front_pass.set_uniform_3fv("_bounds_min", 1, tex_min_bound.vec_array);
        _program_obj_front_pass.set_uniform_3fv("_bounds_max", 1, tex_max_bound.vec_array);

        if (params._cp_enabled) {
            _program_obj_front_pass.set_uniform_3fv("_cross_plane_positions", 1, params._cross_plane_positions.vec_array);
        }
        else {
            _program_obj_front_pass.set_uniform_3fv("_cross_plane_positions", 1, neutral_cross_plane_pos.vec_array);
        }

        _cube.render(GL_FRONT);
        _program_obj_front_pass.unbind();
        // end front pass
#if 1
        if (_do_inside_pass) {
            // back pass
            _program_obj_back_pass.bind();

            _program_obj_back_pass.set_uniform_4fv("_near_plane", 1, &near_plane);

            _program_obj_back_pass.set_uniform_1i("_volume", 0);
            _program_obj_back_pass.set_uniform_1i("_color_alpha", 1);
            _program_obj_back_pass.set_uniform_1i("_geom_depth", 2);

            _program_obj_back_pass.set_uniform_1f("_step_size", 1.0f/params._step_size);
            _program_obj_back_pass.set_uniform_1i("_max_loop_counter", 255);

            _program_obj_back_pass.set_uniform_2fv("_screen_dimensions", 1, params._screen_dimensions.vec_array);

            _program_obj_back_pass.set_uniform_3fv("_bounds_min", 1, tex_min_bound.vec_array);
            _program_obj_back_pass.set_uniform_3fv("_bounds_max", 1, tex_max_bound.vec_array);

            if (params._cp_enabled) {
                _program_obj_back_pass.set_uniform_3fv("_cross_plane_positions", 1, params._cross_plane_positions.vec_array);
            }
            else {
                _program_obj_back_pass.set_uniform_3fv("_cross_plane_positions", 1, neutral_cross_plane_pos.vec_array);
            }

            _cube.render(GL_BACK);
            _program_obj_back_pass.unbind();
            // end back pass
        }
#endif
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
        glDisable(GL_TEXTURE_RECTANGLE_ARB);

        glActiveTexture(GL_TEXTURE1);
        params._color_alpha_texture.unbind();
        
        glActiveTexture(GL_TEXTURE0);
        params._volume_texture.unbind();

        glDisable(GL_BLEND);
        //glEnable(GL_MULTISAMPLE);

#endif
        glMatrixMode(GL_TEXTURE);
        glPopMatrix();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
#else
        // draw transfer function
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);
            params._color_alpha_texture.bind();

                glBegin(GL_QUADS);
                glTexCoord1f(0);
                glVertex2f(0,0);
                glVertex2f(1,0);
                glTexCoord1f(1);
                glVertex2f(1,1);
                glVertex2f(0,1);
                glEnd();

            params._color_alpha_texture.unbind();
        glDisable(GL_BLEND);
#endif 

    }

    bool volume_renderer_raycast_glsl::shutdown()
    {
        return (true);
    }

} // namespace gl
