
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_WAVEFRONT_OBJ_TO_VERTEX_ARRAY_H_INCLUDED
#define SCM_GL_UTIL_WAVEFRONT_OBJ_TO_VERTEX_ARRAY_H_INCLUDED

#include <boost/shared_array.hpp>

#include <scm/core/numeric_types.h>
#include <scm/core/math/math.h>

#include <scm/gl_util/primitives/util/wavefront_obj_file.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {
namespace util {

struct aabbox
{
    scm::math::vec3f    _min;
    scm::math::vec3f    _max;
};

struct vertexbuffer_data
{
    typedef std::vector<std::size_t>                            index_counts_container;
    typedef std::vector<boost::shared_array<scm::uint32> >      index_array_container;
    typedef std::vector<wavefront_material>                     material_container;
    typedef std::vector<aabbox>                                 bbox_container;

    vertexbuffer_data() 
     :  _vert_array_count(0),
        _normals_offset(0),
        _texcoords_offset(0) {}

    boost::shared_array<float>      _vert_array;
    std::size_t                     _vert_array_count;
    std::size_t                     _normals_offset;
    std::size_t                     _texcoords_offset;
    index_array_container           _index_arrays;
    index_counts_container          _index_array_counts;
    material_container              _materials;
    bbox_container                  _bboxes;
}; // struct vertexbuffer_data

// offsets are array offsets in the vertex array, NO byte offsets!
bool __scm_export(gl_util) generate_vertex_buffer(const wavefront_model&               /*in_obj*/,
                                               vertexbuffer_data&                   /*out_data*/,
                                               bool                                 /*interleave_arrays*/ = false);

} // namespace util
} // namespace gl
} // namespace scm

#endif // SCM_GL_UTIL_WAVEFRONT_OBJ_TO_VERTEX_ARRAY_H_INCLUDED
