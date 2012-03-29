
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "axes_compass.h"

#include <scm/gl_classic/opengl.h>

#include <scm/core/math/math.h>
#include <scm/gl_core/math/math.h>

namespace scm {
namespace gl_classic {

axes_compass::axes_compass()
{
}

axes_compass::~axes_compass()
{
}

void axes_compass::render() const
{
    using namespace scm::math;

    mat4f   mv_matrix;

    glGetFloatv(GL_MODELVIEW_MATRIX, mv_matrix.data_array);

    mv_matrix = inverse(mv_matrix);
    
    
    vec3f x_axis = vec3f(mv_matrix[0], mv_matrix[4], mv_matrix[8]);
    vec3f y_axis = vec3f(mv_matrix[1], mv_matrix[5], mv_matrix[9]);
    vec3f z_axis = vec3f(mv_matrix[2], mv_matrix[6], mv_matrix[10]);

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
            glVertex3fv(x_axis.data_array);

            glColor3f( 0, 1, 0);
            glVertex3f(0, 0, 0);
            glVertex3fv(y_axis.data_array);

            glColor3f( 0, 0, 1);
            glVertex3f(0, 0, 0);
            glVertex3fv(z_axis.data_array);
            glEnd();

        glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);

    glLineWidth(1.0f);

    glViewport(cur_viewport[0], cur_viewport[1], cur_viewport[2], cur_viewport[3]);
}

} // namespace gl_classic
} // namespace scm
