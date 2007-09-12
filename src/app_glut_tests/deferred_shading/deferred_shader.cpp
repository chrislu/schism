
#include "deferred_shader.h"

#include <iostream>

#include <scm/ogl/gl.h>
#include <scm/ogl/shader_objects/shader_object.h>

// some utility functionality
namespace {

void draw_texture(unsigned tex_id,
                  const math::vec2ui_t tex_dim,
                  const math::vec2ui_t& ll,
                  const math::vec2ui_t& ur)
{
    // retrieve current matrix mode
    GLint  current_matrix_mode;
    glGetIntegerv(GL_MATRIX_MODE, &current_matrix_mode);

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(ll.x, ll.y, ur.x, ur.y);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glEnable(GL_TEXTURE_RECTANGLE_ARB);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex_id);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(  0.0f, 0.0f);
        glTexCoord2f(float(tex_dim.x), 0.0f);
        glVertex2f(  1.0f, 0.0f);
        glTexCoord2f(float(tex_dim.x), float(tex_dim.y));
        glVertex2f(  1.0f, 1.0f);
        glTexCoord2f(0.0f, float(tex_dim.y));
        glVertex2f(  0.0f, 1.0f);
    glEnd();


    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
    glDisable(GL_TEXTURE_RECTANGLE_ARB);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    glPopAttrib();

    // restore saved matrix mode
    glMatrixMode(current_matrix_mode);

}

} // namespace

namespace scm {

deferred_shader::deferred_shader(unsigned width,
                                 unsigned height)
  : _viewport_dim(width, height)
{
    _framebuffer.reset(new ds_framebuffer(width, height));

    if (!init_shader_programs()) {
        std::cout << "this is when RAII goes wrong..." << std::endl;
        return;
    }
}

deferred_shader::~deferred_shader()
{
    cleanup();
}

bool deferred_shader::init_shader_programs()
{
    boost::scoped_ptr<scm::gl::shader_object>   vert_shader;
    boost::scoped_ptr<scm::gl::shader_object>   frag_shader;


    _fbo_fill_program.reset(new scm::gl::program_object());

    vert_shader.reset(new scm::gl::shader_object(GL_VERTEX_SHADER));
    frag_shader.reset(new scm::gl::shader_object(GL_FRAGMENT_SHADER));

    std::cout << "loading and compiling fill shader:" << std::endl;

    // load shader code from files
    if (!vert_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/fbo_ds_fill_vert.glsl")) {
        std::cout << "Error loading vertex shader:" << std::endl;
        return (false);
    }
    if (!frag_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/fbo_ds_fill_frag.glsl")) {
        std::cout << "Error loading frament shader:" << std::endl;
        return (false);
    }

    // compile shaders
    if (!vert_shader->compile()) {
        std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
        std::cout << vert_shader->get_compiler_output() << std::endl;
        return (false);
    }
    if (!frag_shader->compile()) {
        std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
        std::cout << frag_shader->get_compiler_output() << std::endl;
        return (false);
    }

    // attatch shaders to program object
    if (!_fbo_fill_program->attach_shader(*vert_shader)) {
        std::cout << "unable to attach vertex shader to program object:" << std::endl;
        return (false);
    }
    if (!_fbo_fill_program->attach_shader(*frag_shader)) {
        std::cout << "unable to attach fragment shader to program object:" << std::endl;
        return (false);
    }

    // link program object
    if (!_fbo_fill_program->link()) {
        std::cout << "Error linking program - linker output:" << std::endl;
        std::cout << _fbo_fill_program->get_linker_output() << std::endl;
       return (false);
    }

    _dshading_program.reset(new scm::gl::program_object());

    vert_shader.reset(new scm::gl::shader_object(GL_VERTEX_SHADER));
    frag_shader.reset(new scm::gl::shader_object(GL_FRAGMENT_SHADER));

    std::cout << "loading and compiling deferred shading shader:" << std::endl;

    // load shader code from files
    if (!vert_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/fbo_ds_light_vert.glsl")) {
        std::cout << "Error loadong vertex shader:" << std::endl;
        return (false);
    }
    if (!frag_shader->set_source_code_from_file("./../../../src/app_glut_tests/shader/fbo_ds_light_frag.glsl")) {
        std::cout << "Error loadong frament shader:" << std::endl;
        return (false);
    }

    // compile shaders
    if (!vert_shader->compile()) {
        std::cout << "Error compiling vertex shader - compiler output:" << std::endl;
        std::cout << vert_shader->get_compiler_output() << std::endl;
        return (false);
    }
    if (!frag_shader->compile()) {
        std::cout << "Error compiling fragment shader - compiler output:" << std::endl;
        std::cout << frag_shader->get_compiler_output() << std::endl;
        return (false);
    }

    // attatch shaders to program object
    if (!_dshading_program->attach_shader(*vert_shader)) {
        std::cout << "unable to attach vertex shader to program object:" << std::endl;
        return (false);
    }
    if (!_dshading_program->attach_shader(*frag_shader)) {
        std::cout << "unable to attach fragment shader to program object:" << std::endl;
        return (false);
    }

    // link program object
    if (!_dshading_program->link()) {
        std::cout << "Error linking program - linker output:" << std::endl;
        std::cout << _dshading_program->get_linker_output() << std::endl;
       return (false);
    }

    return (true);
}

void deferred_shader::cleanup()
{

}

void deferred_shader::start_fill_pass() const
{
    static GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0_EXT,
                                    GL_COLOR_ATTACHMENT1_EXT,
                                    GL_COLOR_ATTACHMENT2_EXT};

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _framebuffer->id());
    glDrawBuffers(3, draw_buffers);

    math::mat_glf_t projection;

    math::get_gl_matrix(GL_PROJECTION_MATRIX, projection);

    _projection_inv = math::inverse(projection);

    // bind shader program to current opengl state
    _fbo_fill_program->bind();
    
    // set program parameters
    // set the sampler parameters to the particular texture unit number

    // WARNING, this assumption is only valid for this demo
    //_fbo_fill_program->set_uniform_1i("_diff_gloss", 0);
    //_fbo_fill_program->set_uniform_1i("_normal", 1);

}

void deferred_shader::end_fill_pass() const
{
    _fbo_fill_program->unbind();

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void deferred_shader::shade() const
{
    // save polygon and depth buffer bit
    // to restore culling and depth mask settings later
    glPushAttrib( GL_DEPTH_BUFFER_BIT
                | GL_COLOR_BUFFER_BIT
                | GL_POLYGON_BIT);

    glDisable(GL_BLEND);


    // push current projection matrix
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    // set ortho projection
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    // switch back to modelview matrix\
    // and push current matrix
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

        // reset current matrix to identity
        glLoadIdentity();

        // setup and enable backface culling
        glFrontFace(GL_CCW);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        // disable depth writes
        glDepthMask(false);

        glActiveTexture(GL_TEXTURE0);
        // enable and bind the depth buffer texture rectangle
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _framebuffer->depth_id());

        glActiveTexture(GL_TEXTURE1);
        // enable and bind the color buffer texture rectangle
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _framebuffer->color_id());

        glActiveTexture(GL_TEXTURE2);
        // enable and bind the color buffer texture rectangle
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _framebuffer->specular_id());

        glActiveTexture(GL_TEXTURE3);
        // enable and bind the normal buffer texture rectangle
        glEnable(GL_TEXTURE_RECTANGLE_ARB);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _framebuffer->normal_id());

        _dshading_program->bind();

        _dshading_program->set_uniform_1i("_depth",                 0);
        _dshading_program->set_uniform_1i("_color_gloss",           1);
        _dshading_program->set_uniform_1i("_specular_shininess",    2);
        _dshading_program->set_uniform_1i("_normal",                3);

        math::vec2f_t viewdim = math::vec2f_t(float(_viewport_dim.x), float(_viewport_dim.y));
        _dshading_program->set_uniform_2fv("_viewport_dim", 1, viewdim.vec_array);

        _dshading_program->set_uniform_matrix_4fv("_projection_inverse", 1, false, _projection_inv.mat_array);

        glBegin(GL_TRIANGLES);
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(  0.0f, 0.0f);

            glTexCoord2f(2.0f*float(_viewport_dim.x), 0.0f);
            glVertex2f(  2.0f, 0.0f);

            glTexCoord2f(0.0f, 2.0f*float(_viewport_dim.y));
            glVertex2f(  0.0f, 2.0f);
        glEnd();

        _dshading_program->unbind();


        glActiveTexture(GL_TEXTURE3);
        // unbind and disable the normal buffer texture rectangle
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
        glDisable(GL_TEXTURE_RECTANGLE_ARB);
        glActiveTexture(GL_TEXTURE2);
        // unbind and disable the specular buffer texture rectangle
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
        glDisable(GL_TEXTURE_RECTANGLE_ARB);
        glActiveTexture(GL_TEXTURE1);
        // unbind and disable the color buffer texture rectangle
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
        glDisable(GL_TEXTURE_RECTANGLE_ARB);
        glActiveTexture(GL_TEXTURE0);
        // unbind and disable the depth buffer texture rectangle
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
        glDisable(GL_TEXTURE_RECTANGLE_ARB);


    // restore the saved modelview matrix
    glPopMatrix();
    // restore the saved projection matrix
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

    // restore the saved polygon and depth buffer bits
    // to reset the culling and depth mask settings
    glPopAttrib();
}

void deferred_shader::display_buffers() const
{
    using math::vec2ui_t;

    vec2ui_t out_dim = vec2ui_t(_viewport_dim.x/2, _viewport_dim.y/2);

    draw_texture(_framebuffer->depth_id(),      _viewport_dim, vec2ui_t(0, 0),                                  out_dim);
    draw_texture(_framebuffer->color_id(),      _viewport_dim, vec2ui_t(_viewport_dim.x/2, 0),                  out_dim);
    draw_texture(_framebuffer->normal_id(),     _viewport_dim, vec2ui_t(0, _viewport_dim.y/2),                  out_dim);
    draw_texture(_framebuffer->specular_id(),   _viewport_dim, vec2ui_t(_viewport_dim.x/2, _viewport_dim.y/2),  out_dim);
}


} // namespace scm
