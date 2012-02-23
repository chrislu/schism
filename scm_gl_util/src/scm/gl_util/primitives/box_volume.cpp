
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "box_volume.h"

#include <cassert>
#include <exception>
#include <stdexcept>

#include <boost/assign/list_of.hpp>

#include <scm/core/memory.h>

#include <scm/gl_core/render_device.h>
#include <scm/gl_core/buffer_objects/buffer.h>
#include <scm/gl_core/buffer_objects/vertex_array.h>
#include <scm/gl_core/buffer_objects/vertex_format.h>

namespace scm {
namespace gl {

box_volume_geometry::box_volume_geometry(const render_device_ptr& in_device,
                                         const math::vec3f& in_min_vertex,
                                         const math::vec3f& in_max_vertex)
  : box_geometry(in_device, in_min_vertex, in_max_vertex)
{
    using namespace scm::math;

    int num_vertices            = 6 * 4; // 4 vertices per face

    scoped_array<vec3f>             tex(new vec3f[num_vertices]);

    // front face
    int v = 0;
    tex[v] = vec3f(0.0f, 0.0f, 1.0f); ++v;
    tex[v] = vec3f(1.0f, 0.0f, 1.0f); ++v;
    tex[v] = vec3f(1.0f, 1.0f, 1.0f); ++v;
    tex[v] = vec3f(0.0f, 1.0f, 1.0f); ++v;
    // right face                     
    tex[v] = vec3f(1.0f, 0.0f, 1.0f); ++v;
    tex[v] = vec3f(1.0f, 0.0f, 0.0f); ++v;
    tex[v] = vec3f(1.0f, 1.0f, 0.0f); ++v;
    tex[v] = vec3f(1.0f, 1.0f, 1.0f); ++v;
    // back face                      
    tex[v] = vec3f(1.0f, 0.0f, 0.0f); ++v;
    tex[v] = vec3f(0.0f, 0.0f, 0.0f); ++v;
    tex[v] = vec3f(0.0f, 1.0f, 0.0f); ++v;
    tex[v] = vec3f(1.0f, 1.0f, 0.0f); ++v;
    // left face                      
    tex[v] = vec3f(0.0f, 0.0f, 0.0f); ++v;
    tex[v] = vec3f(0.0f, 0.0f, 1.0f); ++v;
    tex[v] = vec3f(0.0f, 1.0f, 1.0f); ++v;
    tex[v] = vec3f(0.0f, 1.0f, 0.0f); ++v;
    // top face                       
    tex[v] = vec3f(0.0f, 1.0f, 0.0f); ++v;
    tex[v] = vec3f(0.0f, 1.0f, 1.0f); ++v;
    tex[v] = vec3f(1.0f, 1.0f, 1.0f); ++v;
    tex[v] = vec3f(1.0f, 1.0f, 0.0f); ++v;
    // bottom face                    
    tex[v] = vec3f(0.0f, 0.0f, 0.0f); ++v;
    tex[v] = vec3f(1.0f, 0.0f, 0.0f); ++v;
    tex[v] = vec3f(1.0f, 0.0f, 1.0f); ++v;
    tex[v] = vec3f(0.0f, 0.0f, 1.0f); ++v;

    assert(v == (6 * 4));

    using namespace scm::gl;
    using boost::assign::list_of;

    _texcoords.reset();
    _vertex_array.reset();

    _texcoords    = in_device->create_buffer(BIND_VERTEX_BUFFER, USAGE_STATIC_DRAW, num_vertices * sizeof(vec3f), tex.get());
    _vertex_array = in_device->create_vertex_array(vertex_format(0, 0, TYPE_VEC3F)  // position
                                                                (1, 1, TYPE_VEC3F)  // normal
                                                                (2, 2, TYPE_VEC3F), // tex_coord
                                                   list_of(_positions)
                                                          (_normals)
                                                          (_texcoords));
}

box_volume_geometry::~box_volume_geometry()
{
}

void
box_volume_geometry::update(const render_context_ptr& in_context,
                     const math::vec3f& in_min_vertex,
                     const math::vec3f& in_max_vertex)
{
    throw std::runtime_error("box_volume_geometry::update() not implemented");
}

} // namespace gl
} // namespace scm
