
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "volume_crossplanes.h"

#include <scm/gl_classic.h>

#include <scm/core/math/math.h>

namespace scm {
namespace gl_classic {

volume_crossplanes::volume_crossplanes()
    : _planes_dlist(0),
      _planes_dlist_dirty(true),
      _slice_x(0.25f),
      _slice_x_enabled(true),
      _slice_y(0.25f),
      _slice_y_enabled(true),
      _slice_z(0.25f),
      _slice_z_enabled(true),
      _initialized(false)
{
}

volume_crossplanes::~volume_crossplanes()
{
    this->clean_up();
}

void volume_crossplanes::set_slice_x(float x)
{
    _slice_x = math::clamp(x, 0.0f, 1.0f);

    _slice_x_enabled = (x < 0.0f) ? false : true;

    _planes_dlist_dirty = true;
}

void volume_crossplanes::set_slice_y(float y)
{
    _slice_y = math::clamp(y, 0.0f, 1.0f);

    _slice_y_enabled = (y < 0.0f) ? false : true;

    _planes_dlist_dirty = true;
}

void volume_crossplanes::set_slice_z(float z)
{
    _slice_z = math::clamp(z, 0.0f, 1.0f);

    _slice_z_enabled = (z < 0.0f) ? false : true;

    _planes_dlist_dirty = true;
}

bool volume_crossplanes::initialize()
{
    if (_initialized)
        return (true);

    _planes_dlist = glGenLists(1);

    if (_planes_dlist = 0) {
        return (false);
    }

    _initialized = true;
    return (true);
}

void volume_crossplanes::render() const
{
    //if (_planes_dlist_dirty) {
    //    glNewList(_planes_dlist, GL_COMPILE_AND_EXECUTE);
            glPushMatrix();

            glTranslatef(-.5f, -.5f, -.5f);

            if (_slice_z_enabled) {
                // xy-plane
                glBegin(GL_TRIANGLE_STRIP);
                    glTexCoord3f(0.f, 0.f, _slice_z);
                    glVertex3f(  0.f, 0.f, _slice_z);

                    glTexCoord3f(1.f, 0.f, _slice_z);
                    glVertex3f(  1.f, 0.f, _slice_z);

                    glTexCoord3f(0.f, 1.f, _slice_z);
                    glVertex3f(  0.f, 1.f, _slice_z);

                    glTexCoord3f(1.f, 1.f, _slice_z);
                    glVertex3f(  1.f, 1.f, _slice_z);
                glEnd();
            }

            if (_slice_y_enabled) {
                // xz-plane
                glBegin(GL_TRIANGLE_STRIP);
                    glTexCoord3f(0.f, _slice_y, 0.f);
                    glVertex3f(  0.f, _slice_y, 0.f);

                    glTexCoord3f(1.f, _slice_y, 0.f);
                    glVertex3f(  1.f, _slice_y, 0.f);

                    glTexCoord3f(0.f, _slice_y, 1.f);
                    glVertex3f(  0.f, _slice_y, 1.f);

                    glTexCoord3f(1.f, _slice_y, 1.f);
                    glVertex3f(  1.f, _slice_y, 1.f);
                glEnd();
            }

            if (_slice_x_enabled) {
                // yz-plane
                glBegin(GL_TRIANGLE_STRIP);
                    glTexCoord3f(_slice_x, 0.f, 0.f);
                    glVertex3f(  _slice_x, 0.f, 0.f);

                    glTexCoord3f(_slice_x, 1.f, 0.f);
                    glVertex3f(  _slice_x, 1.f, 0.f);

                    glTexCoord3f(_slice_x, 0.f, 1.f);
                    glVertex3f(  _slice_x, 0.f, 1.f);

                    glTexCoord3f(_slice_x, 1.f, 1.f);
                    glVertex3f(  _slice_x, 1.f, 1.f);
                glEnd();
            }

            glPopMatrix();
    //    glEndList();

    //    _planes_dlist_dirty = false;
    //}
    //else {
    //    glCallList(_planes_dlist);
    //}
}

void volume_crossplanes::clean_up()
{
}

} // namespace gl_classic
} // namespace scm
