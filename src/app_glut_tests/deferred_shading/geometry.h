
#ifndef APP_BVR_GEOMETRY_H_INCLUDED
#define APP_BVR_GEOMETRY_H_INCLUDED

#include <vector>
#include <boost/shared_ptr.hpp>

#include <scm/data/geometry/scm_geom/scm_geom.h>
#include <scm/data/geometry/wavefront_obj/obj_file.h>
#include <scm/ogl/vertexbuffer_object/vertexbuffer.h>
#include <scm/ogl/vertexbuffer_object/indexbuffer.h>

struct geometry
{
    typedef std::vector<boost::shared_ptr<scm::gl::indexbuffer> >   index_buffer_container;
    typedef std::vector<scm::data::wavefront_material>              material_container;

    scm::data::geometry_descriptor                  _desc;

    boost::shared_ptr<scm::gl::vertexbuffer>        _vbo;
    index_buffer_container                          _indices;
    material_container                              _materials;

    unsigned                                        _face_count;

}; // geometry

extern std::vector<geometry>        _geometries;

bool open_geometry_file(const std::string& filename);
bool open_geometry();

#endif // APP_BVR_GEOMETRY_H_INCLUDED
