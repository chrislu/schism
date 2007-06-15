
#include "obj_to_vertex_array.h"

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
                            boost::shared_array<float>&          out_vert_array,
                            std::size_t&                         vert_array_count,
                            std::size_t&                         normals_offset,
                            std::size_t&                         texcoords_offset,
                            boost::shared_array<core::uint32_t>& out_trilist_index_array,
                            std::size_t&                         index_array_count,
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
    unsigned new_index      = 0;
    unsigned iarray_index   = 0;

    foreach (const wavefront_object& wf_obj, in_obj._objects) {
        foreach (const wavefront_object_group& wf_obj_grp, wf_obj._groups) {
            index_buf_size += 3 * static_cast<unsigned>(wf_obj_grp._num_tri_faces);
        }
    }

    index_array_count   = index_buf_size;
    out_trilist_index_array.reset(new core::uint32_t[index_buf_size]);

    foreach (const wavefront_object& wf_obj, in_obj._objects) {
        foreach (const wavefront_object_group& wf_obj_grp, wf_obj._groups) {
            for (unsigned i = 0; i < wf_obj_grp._num_tri_faces; ++i) {
                const wavefront_object_triangle_face& cur_face = wf_obj_grp._tri_faces[i];

                for (unsigned k = 0; k < 3; ++k) {
                    obj_vert_index  cur_index(cur_face._vertices[k],
                                              in_obj._num_tex_coords != 0 ? cur_face._tex_coords[k] : 0,
                                              in_obj._num_normals    != 0 ? cur_face._normals[k] : 0);
                    
                    index_mapping::const_iterator prev_it = indices.find(cur_index);

                    if (prev_it == indices.end()) {
                        indices.insert(index_value(cur_index, new_index));
                        out_trilist_index_array[iarray_index] = new_index;
                        ++new_index;
                    }
                    else {
                        out_trilist_index_array[iarray_index] = prev_it->second;
                    }

                    ++iarray_index;
                }
            }
        }
    }

    // second pass
    // copy vertex data according to new indices
    std::size_t     array_size = 3 * indices.size();
    if (in_obj._num_normals != 0) {
        array_size          += 3 * indices.size();
        normals_offset      = 3 * indices.size();
    }
    else {
        normals_offset      = 0;
    }

    if (in_obj._num_tex_coords != 0) {
        array_size          += 2 * indices.size();
        texcoords_offset    = normals_offset + 3 * indices.size();
    }
    else {
        texcoords_offset    = 0;
    }

    vert_array_count    = indices.size();
    out_vert_array.reset(new float[array_size]);

    unsigned varray_index = 0;

    for (index_mapping::const_iterator ind_it = indices.begin();
         ind_it != indices.end();
         ++ind_it) {

        const obj_vert_index&  cur_index = ind_it->first;

        varray_index =  ind_it->second;

        memcpy(out_vert_array.get() + varray_index * 3,
               in_obj._vertices[cur_index._v - 1].vec_array,
               3 * sizeof(float));

        if (normals_offset) {
            memcpy(out_vert_array.get() + varray_index * 3
                                        + normals_offset,
                   in_obj._normals[cur_index._n - 1].vec_array,
                   3 * sizeof(float));
        }
        if (texcoords_offset) {
            memcpy(out_vert_array.get() + varray_index * 2
                                        + texcoords_offset,
                   in_obj._tex_coords[cur_index._t - 1].vec_array,
                   2 * sizeof(float));
        }
    }

    return (true);
}

} // namespace data
} // namespace scm
