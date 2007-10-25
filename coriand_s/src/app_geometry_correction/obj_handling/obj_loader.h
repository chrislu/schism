
#ifndef OBJ_LOADER_H_INCLUDED
#define OBJ_LOADER_H_INCLUDED

#include <string>

#include <obj_handling/obj_file.h>

namespace scm
{
    bool open_obj_file(const std::string& filename, scm::wavefront_model& out_obj);

} // namespace scm

#endif // OBJ_LOADER_H_INCLUDED



