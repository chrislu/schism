
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.


#include <scm/gl_classic.h>
#include <scm/gl_classic/opengl.h>

#include "volume_textured_unit_cube.h"

namespace scm {
namespace gl_classic {

volume_textured_unit_cube::volume_textured_unit_cube()
    : _vertices_vbo(0),
      _indices_vbo(0),
      _initialized(false)
{
}

volume_textured_unit_cube::~volume_textured_unit_cube()
{
    this->clean_up();
}

bool volume_textured_unit_cube::initialize()
{
    if (_initialized) {
        return (true);
    }

    //if (!scm::opengl::get().is_supported("GL_VERSION_1_5")) {
    //    return (false);
    //}

    struct vertex_format
    {
        float       pos[3];
        float       tex[3];
    };

    vertex_format* vertices = new vertex_format[8];
    // trianglestrip
    unsigned short* indices = new unsigned short[36];
    // triangles
    //unsigned short* indices = new unsigned short[36];

    // generate vertices for a unit cube
    // for triangles
    //for (unsigned int v = 0; v < 8; v++) {
    //    vertices[v].pos[0] = vertices[v].tex[0] = (float)(v & 0x01);
    //    vertices[v].pos[1] = vertices[v].tex[1] = (float)((v & 0x02) >> 1);
    //    vertices[v].pos[2] = vertices[v].tex[2] = (float)((v & 0x04) >> 2);
    //}
    // for quads
    for (unsigned int v = 0; v < 8; v++) {
        vertices[v].pos[2] = vertices[v].tex[2] = (float)(v & 0x01);
        vertices[v].pos[1] = vertices[v].tex[1] = (float)((v & 0x02) >> 1);
        vertices[v].pos[0] = vertices[v].tex[0] = (float)((v & 0x04) >> 2);
    }

    // quads
    indices[0]  = 1;
    indices[1]  = 5;
    indices[2]  = 7;
    indices[3]  = 3;

    indices[4]  = 5;
    indices[5]  = 4;
    indices[6]  = 6;
    indices[7]  = 7;

    indices[8]  = 7;
    indices[9]  = 6;
    indices[10] = 2;
    indices[11] = 3;

    indices[12] = 3;
    indices[13] = 2;
    indices[14] = 0;
    indices[15] = 1;

    indices[16] = 1;
    indices[17] = 0;
    indices[18] = 4;
    indices[19] = 5;

    indices[20] = 0;
    indices[21] = 2;
    indices[22] = 6;
    indices[23] = 4;


    // trianglestrip
    //indices[0]  = 4;
    //indices[1]  = 5;
    //indices[2]  = 6;
    //indices[3]  = 7;
    //indices[4]  = 3;
    //indices[5]  = 5;
    //indices[6]  = 1;
    //indices[7]  = 4;
    //indices[8]  = 0;
    //indices[9]  = 6;
    //indices[10] = 2;
    //indices[11] = 3;
    //indices[12] = 0;
    //indices[13] = 1;
    // triangles
    //indices[0]  = 4;
    //indices[1]  = 5;
    //indices[2]  = 6;
    //indices[3]  = 6;
    //indices[4]  = 5;
    //indices[5]  = 7;

    //indices[6]  = 6;
    //indices[7]  = 7;
    //indices[8]  = 2;
    //indices[9]  = 2;
    //indices[10] = 7;
    //indices[11] = 3;

    //indices[12] = 2;
    //indices[13] = 3;
    //indices[14] = 0;
    //indices[15] = 0;
    //indices[16] = 3;
    //indices[17] = 1;

    //indices[18] = 0;
    //indices[19] = 1;
    //indices[20] = 4;
    //indices[21] = 4;
    //indices[22] = 1;
    //indices[23] = 5;

    //indices[24] = 5;
    //indices[25] = 1;
    //indices[26] = 7;
    //indices[27] = 7;
    //indices[28] = 1;
    //indices[29] = 3;

    //indices[30] = 0;
    //indices[31] = 4;
    //indices[32] = 2;
    //indices[33] = 2;
    //indices[34] = 4;
    //indices[35] = 6;

    glGenBuffers(1, &_vertices_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vertices_vbo);
    glBufferData(GL_ARRAY_BUFFER, 8*sizeof(vertex_format), vertices, GL_STATIC_DRAW);
    
    if (glGetError() != GL_NONE) {
        clean_up();
        delete [] vertices;
        delete [] indices;
        return (false);
    }

    glGenBuffers(1, &_indices_vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indices_vbo);
    // quads
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 24*sizeof(unsigned short), indices, GL_STATIC_DRAW);
    // trianglestrip
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, 14*sizeof(unsigned short), indices, GL_STATIC_DRAW);
    // triangles
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36*sizeof(unsigned short), indices, GL_STATIC_DRAW);

    if (glGetError() != GL_NONE) {
        clean_up();
        delete [] vertices;
        delete [] indices;
        return (false);
    }

    _initialized = true;
    
    delete [] vertices;
    delete [] indices;

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    return (true);
}

void volume_textured_unit_cube::render(int draw_face) const
{
    glPushAttrib(GL_POLYGON_BIT);

    glFrontFace(GL_CCW);


    switch (draw_face) {
        case GL_FRONT:  glEnable(GL_CULL_FACE);
                        glCullFace(GL_BACK);break;
        case GL_BACK:   glEnable(GL_CULL_FACE);
                        glCullFace(GL_FRONT);break;
        case GL_FRONT_AND_BACK:
        default:        /*glDisable(GL_CULL_FACE);*/break;

    };

    glBindBuffer(GL_ARRAY_BUFFER, _vertices_vbo);
    glVertexPointer(3, GL_FLOAT, 6*sizeof(float), 0);
    glTexCoordPointer(3, GL_FLOAT, 6*sizeof(float), (GLvoid*)(0 + 3*sizeof(float)));

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indices_vbo);
    
    // quads
    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_SHORT, NULL);
    // trianglestrip
    //glDrawElements(GL_TRIANGLE_STRIP, 14, GL_UNSIGNED_SHORT, NULL);
    // triangles
    //glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, NULL);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glPopAttrib();
}

void volume_textured_unit_cube::clean_up()
{
    glDeleteBuffers(1, &_vertices_vbo);
    glDeleteBuffers(1, &_indices_vbo);
}

} // namespace gl_classic
} // namespace scm
