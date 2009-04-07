
#ifndef SCM_DATA_OBJ_LOADER_H_INCLUDED
#define SCM_DATA_OBJ_LOADER_H_INCLUDED

#include <string>

#include <scm/data/geometry/wavefront_obj/obj_file.h>

#include <scm/core/platform/platform.h>

namespace scm {
namespace data {
    
bool __scm_export(data) open_obj_file(const std::string& filename, wavefront_model& /*out_obj*/);

} // namespace data
} // namespace scm

#endif // SCM_DATA_OBJ_LOADER_H_INCLUDED
