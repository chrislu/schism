
#ifndef GLUT_TESTS_DEFERRED_SHADER_H_INCLUDED
#define GLUT_TESTS_DEFERRED_SHADER_H_INCLUDED

#include <boost/scoped_ptr.hpp>

#include <deferred_shading/framebuffer.h>

#include <scm/core/math/math.h>
#include <scm/core/math/math_gl.h>
#include <scm/ogl/shader_objects/program_object.h>

namespace scm {

class deferred_shader
{
public:
    deferred_shader(unsigned width,
                    unsigned height);
    virtual ~deferred_shader();


    void                    start_fill_pass() const;
    void                    end_fill_pass() const;

    void                    shade() const;
    void                    display_buffers() const;

protected:

private:
    boost::scoped_ptr<ds_framebuffer>           _framebuffer;

    boost::scoped_ptr<scm::gl::program_object>  _fbo_fill_program;
    boost::scoped_ptr<scm::gl::program_object>  _dshading_program;

    math::vec2ui_t          _viewport_dim;
    mutable math::mat_glf_t _projection_inv;

    bool                    init_shader_programs();
    void                    cleanup();

}; // class deferred_shader

} // namespace scm

#endif // GLUT_TESTS_DEFERRED_SHADER_H_INCLUDED
