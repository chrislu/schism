
#ifndef OBJ_FILE_H_INCLUDED
#define OBJ_FILE_H_INCLUDED

#include <cstdlib>

#include <string>
#include <vector>

#include <scm/core/ptr_types.h>
#include <scm/core/math/math.h>

namespace scm
{
    struct triangle_face
    {
        unsigned    _vertices[3];
        unsigned    _normals[3];
        unsigned    _tex_coords[3];        
    };

    struct wavefront_object_group
    {
        wavefront_object_group() : _num_tri_faces(0) {}

        std::size_t                                 _num_tri_faces;
        scm::core::shared_array<scm::triangle_face> _tri_faces;

        std::string                                 _name;

    }; // wavefront_object_group

    struct wavefront_object
    {
        wavefront_object() {}

        typedef std::vector<wavefront_object_group> group_container;
        group_container                             _groups;

        std::string                                 _name;
    }; // struct wavefront_object

    struct wavefront_model
    {
        wavefront_model() : _num_vertices(0),
                            _num_normals(0),
                            _num_tex_coords(0) {}

        typedef std::vector<wavefront_object>       object_container;
        object_container                            _objects;

        std::size_t                                 _num_vertices;
        std::size_t                                 _num_normals;
        std::size_t                                 _num_tex_coords;

        scm::core::shared_array<math::vec3f_t>      _vertices;
        scm::core::shared_array<math::vec3f_t>      _normals;
        scm::core::shared_array<math::vec2f_t>      _tex_coords;


    }; // struct wavefront_model


} // namespace scm

#endif // OBJ_FILE_H_INCLUDED



