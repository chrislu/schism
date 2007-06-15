

#ifndef SCM_DATA_OBJ_TO_VERTEX_ARRAY_H_INCLUDED
#define SCM_DATA_OBJ_TO_VERTEX_ARRAY_H_INCLUDED

#include <boost/shared_array.hpp>

#include <scm/core/int_types.h>
#include <scm/core/math/math.h>
#include <scm/data/geometry/wavefront_obj/obj_file.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace data {
   
// offsets are array offsets in the vertex array, NO byte offsets!
bool __scm_export(data) generate_vertex_buffer(const wavefront_model&               /*in_obj*/,
                                               boost::shared_array<float>&          /*out_vert_array*/,
                                               std::size_t&                         /*vert_array_count*/,
                                               std::size_t&                         /*normals_offset*/,
                                               std::size_t&                         /*texcoords_offset*/,
                                               boost::shared_array<core::uint32_t>& /*out_trilist_index_array*/,
                                               std::size_t&                         /*index_array_count*/,
                                               bool                                 /*interleave_arrays*/ = false);

} // namespace data
} // namespace scm

#endif // SCM_DATA_OBJ_TO_VERTEX_ARRAY_H_INCLUDED
