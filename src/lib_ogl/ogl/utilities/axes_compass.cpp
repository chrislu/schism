
#include "axes_compass.h"

#include <ogl/gl.h>

#include <scm_core/math/math.h>
#include <scm_core/math/math_gl.h>

namespace gl
{
    axes_compass::axes_compass()
    {
    }

    axes_compass::~axes_compass()
    {
    }

    void axes_compass::render() const
    {
        using namespace math;

        mat_glf_t mv_matrix;
        get_gl_matrix(GL_MODELVIEW_MATRIX, mv_matrix);

        mv_matrix = inverse(mv_matrix);
        
        
        vec3f_t x_axis = vec3f_t(mv_matrix.mat_array[0], mv_matrix.mat_array[4], mv_matrix.mat_array[8]);
        vec3f_t y_axis = vec3f_t(mv_matrix.mat_array[1], mv_matrix.mat_array[5], mv_matrix.mat_array[9]);
        vec3f_t z_axis = vec3f_t(mv_matrix.mat_array[2], mv_matrix.mat_array[6], mv_matrix.mat_array[10]);

        //vec3f_t x_axis = vec3f_t(mv_matrix[0][0], mv_matrix[1][0], mv_matrix[2][0]);
        //vec3f_t y_axis = vec3f_t(mv_matrix[0][1], mv_matrix[1][1], mv_matrix[2][1]);
        //vec3f_t z_axis = vec3f_t(mv_matrix[0][2], mv_matrix[1][2], mv_matrix[2][2]);

        GLsizei cur_viewport[4];
        GLsizei new_viewport[4];
        glGetIntegerv(GL_VIEWPORT, cur_viewport);

        for (unsigned i = 0; i < 4; i++) {
            new_viewport[i] = cur_viewport[i] / 8;
        }

        GLsizei new_vp_dim = math::min(new_viewport[2], new_viewport[3]);

        glViewport(new_viewport[0], new_viewport[1], new_vp_dim, new_vp_dim);

        glLineWidth(1.5f);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glFrustum(-0.5f, 0.5f, -0.5f, 0.5f, 0.7f, 3);

            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
                glLoadIdentity();

                glTranslatef(0,0,-2);
                glBegin(GL_LINES);
                glColor3f( 1, 0, 0);
                glVertex3f(0, 0, 0);
                glVertex3fv(x_axis.vec_array);

                glColor3f( 0, 1, 0);
                glVertex3f(0, 0, 0);
                glVertex3fv(y_axis.vec_array);

                glColor3f( 0, 0, 1);
                glVertex3f(0, 0, 0);
                glVertex3fv(z_axis.vec_array);
                glEnd();

            glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();

        glMatrixMode(GL_MODELVIEW);

        glLineWidth(1.0f);

        glViewport(cur_viewport[0], cur_viewport[1], cur_viewport[2], cur_viewport[3]);
    }

} // namespace gl



