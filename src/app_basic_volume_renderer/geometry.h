
#ifndef APP_BVR_GEOMETRY_H_INCLUDED
#define APP_BVR_GEOMETRY_H_INCLUDED

#include <vector>
#include <boost/shared_ptr.hpp>

#include <scm/data/geometry/scm_geom/scm_geom.h>
#include <scm/ogl/vertexbuffer_object/vertexbuffer_object.h>

struct geometry
{
    scm::data::geometry_descriptor                  _desc;
    boost::shared_ptr<scm::gl::vertexbuffer_object> _vbo;

}; // geometry

extern std::vector<geometry>        _geometries;

bool open_geometry_file(const std::string& filename);
bool open_geometry();

#endif // APP_BVR_GEOMETRY_H_INCLUDED
