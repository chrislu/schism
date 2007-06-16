
#include "volume_renderer.h"

#include <scm/core/math/math.h>
#include <scm/core/math/math_gl.h>

#include <scm/ogl/gl.h>

#include <volume_renderer/volume_renderer_parameters.h>

namespace gl
{
    volume_renderer::volume_renderer()
    {
    }

    volume_renderer::~volume_renderer()
    {
    }

    bool volume_renderer::initialize()
    {
        static bool initialized = false;

        if (initialized) {
            return (true);
        }

        if (!_cube.initialize()) {
            return (false);
        }

        if (!_planes.initialize()) {
            return (false);
        }

        initialized = true;

        return (true);
    }

    void volume_renderer::draw_outlines(const gl::volume_renderer_parameters& params)
    {
        // draw outlines
        glPushMatrix();
        glPushAttrib(GL_POLYGON_BIT | GL_LIGHTING_BIT);

            glDisable(GL_LIGHTING);

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glEnable(GL_POLYGON_OFFSET_LINE);

            glPolygonOffset(-10.0f, -20.0f);
            glPushMatrix();
                glScalef(params._aspect.x,
                         params._aspect.y,
                         params._aspect.z);
                glTranslatef(-0.5f, -0.5f, -0.5f);
                glColor3f(0, 0, 0.7f);
                _cube.render(GL_FRONT);
            glPopMatrix();

            // volume lens outline
            glPolygonOffset(-10.0f, -15.0f);
            glPushMatrix();
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
                glColor3f(0, 0.7f, 0);
                _cube.render(GL_FRONT);
            glPopMatrix();

            // volume outline
            glPolygonOffset(10.0f, 15.0f);
            glPushMatrix();
                glScalef(params._aspect.x,
                         params._aspect.y,
                         params._aspect.z);
                glTranslatef(-0.5f, -0.5f, -0.5f);
                glColor3f(0, 0, 0.7f);
                _cube.render(GL_BACK);
            glPopMatrix();

            // volume lens outline
            glPolygonOffset(10.0f, 20.0f);
            glPushMatrix();
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
                glColor3f(0, 0.7f, 0);
                _cube.render(GL_BACK);
            glPopMatrix();
            glDisable(GL_POLYGON_OFFSET_LINE);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glPopAttrib();
        glPopMatrix();
        // end draw outlines
    }

    void volume_renderer::draw_bounding_volume(const gl::volume_renderer_parameters& params)
    {
        glPushMatrix();

            glTranslatef(params._point_of_interest.x - 0.5f,
                         params._point_of_interest.y - 0.5f,
                         params._point_of_interest.z - 0.5f);
            glScalef(params._extend.x,
                     params._extend.y,
                     params._extend.z);
            glScalef(params._aspect.x,
                     params._aspect.y,
                     params._aspect.z);
            //glTranslatef(-0.5f, -0.5f, -0.5f);

            _cube.render(GL_FRONT_AND_BACK);

        glPopMatrix();
    }

} // namespace gl
