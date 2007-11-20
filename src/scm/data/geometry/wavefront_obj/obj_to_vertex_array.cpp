
#include "obj_to_vertex_array.h"

#include <limits>
#include <map>

#include <scm/core/utilities/foreach.h>

namespace {

struct obj_vert_index
{
    unsigned    _v;
    unsigned    _t;
    unsigned    _n;

    obj_vert_index(unsigned v, unsigned t, unsigned n) : _v(v), _t(t), _n(n) {}
    // lexicographic compare of the index vector
    bool operator<(const obj_vert_index& rhs) const {
        if (_v == rhs._v && _t == rhs._t) return (_n < rhs._n);
        if (_v == rhs._v) return (_t < rhs._t);
        return (_v < rhs._v);
    }

}; // struct obj_vert_index

} // namespace


namespace scm {
namespace data {

bool generate_vertex_buffer(const wavefront_model&               in_obj,
                            vertexbuffer_data&                   out_data,
                            bool                                 interleave_arrays)
{
    if (interleave_arrays) {
        return (false);
    }

    typedef std::map<obj_vert_index, unsigned>      index_mapping;
    typedef index_mapping::value_type               index_value;

    index_mapping       indices;

    wavefront_model::object_container::const_iterator     cur_obj_it;
    wavefront_object::group_container::const_iterator     cur_grp_it;
    unsigned index_buf_size = 0;

    // first pass
    // find out the size of our new arrays and reorder the indices

    out_data._index_arrays.reserve(in_obj._objects.size());
    out_data._index_array_counts.reserve(in_obj._objects.size());

    foreach (const wavefront_object& wf_obj, in_obj._objects) {
        foreach (const wavefront_object_group& wf_obj_grp, wf_obj._groups) {
            out_data._index_array_counts.push_back(3 * static_cast<unsigned>(wf_obj_grp._num_tri_faces));

            wavefront_model::material_container::const_iterator mat = in_obj._materials.find(wf_obj_grp._material_name);

            if (mat != in_obj._materials.end()) {
                out_data._materials.push_back(mat->second);
            }
            else {
                out_data._materials.push_back(wavefront_material());
            }
        }
    }

    vertexbuffer_data::index_counts_container::iterator     cur_index_count = out_data._index_array_counts.begin();

    unsigned new_index      = 0;
    unsigned iarray_index   = 0;

    foreach (const wavefront_object& wf_obj, in_obj._objects) {
        foreach (const wavefront_object_group& wf_obj_grp, wf_obj._groups) {

            iarray_index   = 0;

            // initialize index array
            out_data._index_arrays.push_back(boost::shared_array<core::uint32_t>());
            vertexbuffer_data::index_array_container::value_type& cur_index_array = out_data._index_arrays.back();
            cur_index_array.reset(new core::uint32_t[*cur_index_count]);

            // initialize bbox
            out_data._bboxes.push_back(aabbox());
            vertexbuffer_data::bbox_container::value_type& cur_bbox = out_data._bboxes.back();

            cur_bbox._min   = math::vec3f_t((std::numeric_limits<math::vec3f_t::component_type>::max)());
            cur_bbox._max   = math::vec3f_t((std::numeric_limits<math::vec3f_t::component_type>::min)());

            for (unsigned i = 0; i < wf_obj_grp._num_tri_faces; ++i) {
                const wavefront_object_triangle_face& cur_face = wf_obj_grp._tri_faces[i];

                for (unsigned k = 0; k < 3; ++k) {
                    obj_vert_index  cur_index(cur_face._vertices[k],
                                              in_obj._num_tex_coords != 0 ? cur_face._tex_coords[k] : 0,
                                              in_obj._num_normals    != 0 ? cur_face._normals[k] : 0);
                    
                    // update bounding box
                    const math::vec3f_t& cur_vert = in_obj._vertices[cur_index._v];

                    for (unsigned c = 0; c < 3; ++c) {
                        cur_bbox._min.vec_array[c] = cur_vert.vec_array[c] < cur_bbox._min.vec_array[c] ? cur_vert.vec_array[c] : cur_bbox._min.vec_array[c];
                        cur_bbox._max.vec_array[c] = cur_vert.vec_array[c] > cur_bbox._max.vec_array[c] ? cur_vert.vec_array[c] : cur_bbox._max.vec_array[c];
                    }

                    // check index mapping
                    index_mapping::const_iterator prev_it = indices.find(cur_index);

                    if (prev_it == indices.end()) {
                        indices.insert(index_value(cur_index, new_index));
                        cur_index_array[iarray_index] = new_index;
                        ++new_index;
                    }
                    else {
                        cur_index_array[iarray_index] = prev_it->second;
                    }

                    ++iarray_index;
                }
            }

            ++cur_index_count;
        }
    }

    // second pass
    // copy vertex data according to new indices
    std::size_t     array_size = 3 * indices.size();
    if (in_obj._num_normals != 0) {
        array_size                 += 3 * indices.size();
        out_data._normals_offset    = 3 * indices.size();
    }
    else {
        out_data._normals_offset    = 0;
    }

    if (in_obj._num_tex_coords != 0) {
        array_size                 += 2 * indices.size();
        out_data._texcoords_offset  = out_data._normals_offset + 3 * indices.size();
    }
    else {
        out_data._texcoords_offset  = 0;
    }

    out_data._vert_array_count      = indices.size();
    out_data._vert_array.reset(new float[array_size]);

    unsigned varray_index = 0;

    for (index_mapping::const_iterator ind_it = indices.begin();
         ind_it != indices.end();
         ++ind_it) {

        const obj_vert_index&  cur_index = ind_it->first;

        varray_index =  ind_it->second;

        memcpy(out_data._vert_array.get() + varray_index * 3,
               in_obj._vertices[cur_index._v - 1].vec_array,
               3 * sizeof(float));

        if (out_data._normals_offset) {
            memcpy(out_data._vert_array.get() + varray_index * 3
                                              + out_data._normals_offset,
                   in_obj._normals[cur_index._n - 1].vec_array,
                   3 * sizeof(float));
        }
        if (out_data._texcoords_offset) {
            memcpy(out_data._vert_array.get() + varray_index * 2
                                              + out_data._texcoords_offset,
                   in_obj._tex_coords[cur_index._t - 1].vec_array,
                   2 * sizeof(float));
        }
    }

    return (true);
}

} // namespace data
} // namespace scm
