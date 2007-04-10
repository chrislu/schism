
#include "obj_loader.h"

#include <scm_core/platform/platform.h>

#if    SCM_COMPILER     == SCM_COMPILER_MSVC \
    && SCM_COMPILER_VER >= 1400

#pragma warning(push)               // preserve warning settings
#pragma warning(disable : 4244)     // disable warning C4244: lexical_cast initializing size_t to int

#endif //    SCM_COMPILER     == SCM_COMPILER_MSVC
       // && SCM_COMPILER_VER >= 1400

#include <obj_handling/obj_file.h>
//#include <obj_handling/obj_file_grammar.h>

#include <boost/lexical_cast.hpp>
//#include <boost/spirit.hpp>
//#include <boost/spirit/iterator/file_iterator.hpp>

#include <cassert>
#include <fstream>
#include <string>
#include <strstream>
#include <vector>

namespace scm
{

bool open_obj_file(const std::string& filename, scm::wavefront_model& out_obj)
{
    std::ifstream   obj_file;

    obj_file.open(filename.c_str(), std::ios_base::in);

    if (!obj_file) {
        return (false);
    }

    if (!out_obj._objects.empty()) {
        // clear model structure
        out_obj._objects.clear();
    }

    std::string     cur_line;
    bool            primitive_definition_started = false;

    wavefront_model::object_container::iterator     cur_obj_it;
    wavefront_object::group_container::iterator     cur_grp_it;

    // first pass trough the file
    // collect data about file
    while (std::getline(obj_file, cur_line)) {
        
        std::string::const_iterator b = cur_line.begin();
        std::string::const_iterator e = cur_line.end();

        std::istrstream line(cur_line.c_str());

        char line_id;
        line.get(line_id);

        switch (line_id) {
            case 'v': {
                    if (primitive_definition_started || out_obj._objects.empty()) {
                        // start new object
                        out_obj._objects.push_back(wavefront_object());
                        cur_obj_it = out_obj._objects.end() - 1;
                        cur_obj_it->_name = std::string("object_") + boost::lexical_cast<std::string>(out_obj._objects.size() - 1);

                        primitive_definition_started = false;
                    }

                    line.get(line_id);
                    switch (line_id) {
                        case ' ': ++out_obj._num_vertices;break;
                        case 'n': ++out_obj._num_normals;break;
                        case 't': ++out_obj._num_tex_coords;break;
                    }
                }
                break;
            case 'g': {
                    cur_obj_it->_groups.push_back(wavefront_object_group());
                    cur_grp_it = cur_obj_it->_groups.end() - 1;

                    primitive_definition_started = true;
                    cur_grp_it->_name = std::string(b + 2, e);
                }
                break;
            case 'o': {
                    out_obj._objects.push_back(wavefront_object());
                    cur_obj_it = out_obj._objects.end() - 1;

                    primitive_definition_started = false;
                    cur_obj_it->_name = std::string(b + 2, e);
                }
                break;
            case 'f': {
                    if (!primitive_definition_started) {
                        // start new group in current object
                        cur_obj_it->_groups.push_back(wavefront_object_group());
                        cur_grp_it = cur_obj_it->_groups.end() - 1;
                        cur_grp_it->_name = std::string("group_") + boost::lexical_cast<std::string>(cur_obj_it->_groups.size() - 1);
                    }
                    primitive_definition_started = true;
                    ++cur_grp_it->_num_tri_faces;
                }
                break;
            case '#':break;
            default:;
        }
    }


    if (out_obj._objects.empty()) {
        obj_file.close();

        return (false);
    }
    else {
        // initialize wavefront_model structure
        if (out_obj._num_vertices != 0) {
            out_obj._vertices.reset(new math::vec3f_t[out_obj._num_vertices]);
        }
        if (out_obj._num_normals != 0) {
            out_obj._normals.reset(new math::vec3f_t[out_obj._num_normals]);
        }
        if (out_obj._num_tex_coords != 0) {
            out_obj._tex_coords.reset(new math::vec2f_t[out_obj._num_tex_coords]);
        }

        for (cur_obj_it = out_obj._objects.begin(); cur_obj_it != out_obj._objects.end(); ++cur_obj_it) {
            for (cur_grp_it = cur_obj_it->_groups.begin(); cur_grp_it != cur_obj_it->_groups.end(); ++cur_grp_it) {
                if (cur_grp_it->_num_tri_faces != 0) {
                    cur_grp_it->_tri_faces.reset(new scm::triangle_face[cur_grp_it->_num_tri_faces]);
                }
            }
        }
    }


    // second pass trough the file
    // this time around we know what is to expect in there

    unsigned next_vertex_index      = 0;
    unsigned next_normal_index      = 0;
    unsigned next_tex_coord_index   = 0;

    unsigned next_face_index        = 0;

    cur_obj_it = out_obj._objects.begin();
    cur_grp_it = cur_obj_it->_groups.begin();

    primitive_definition_started = false;

    obj_file.clear();
    obj_file.seekg(0);

    while (std::getline(obj_file, cur_line)) {
        std::istrstream line(cur_line.c_str());

        char line_id;
        line.get(line_id);

        switch (line_id) {
            case 'v': {
                    if (primitive_definition_started) {
                        // start of next object
                        ++cur_obj_it;
                        next_face_index = 0;
                        primitive_definition_started = false;
                    }
                    line.get(line_id);
                    switch (line_id) {
                        case ' ': {
                                math::vec3f_t&  v = out_obj._vertices[next_vertex_index++];
                                line >> v.x;
                                line >> v.y;
                                line >> v.z;
                            }
                            break;
                        case 'n': {
                                math::vec3f_t&  n = out_obj._normals[next_normal_index++];
                                line >> n.x;
                                line >> n.y;
                                line >> n.z;
                            }
                            break;
                        case 't': {
                                math::vec2f_t&  t = out_obj._tex_coords[next_tex_coord_index++];
                                line >> t.x;
                                line >> t.y;
                            }
                            break;
                    }
                }
                break;
            case 'g': {
                    if (!primitive_definition_started) {
                        cur_grp_it = cur_obj_it->_groups.begin();
                    }
                    else {
                        // start of next group in current object
                        ++cur_grp_it;
                    }
                    primitive_definition_started = true;
                }
                break;
            case 'o': {
                    if (primitive_definition_started) {
                        // start of next object
                        ++cur_obj_it;
                        next_face_index = 0;
                        primitive_definition_started = false;
                    }
                }
                break;
            case 'f': {
                    if (!primitive_definition_started) {
                        cur_grp_it = cur_obj_it->_groups.begin();
                        next_face_index = 0;
                    }
                    primitive_definition_started = true;

                    scm::triangle_face& t = cur_grp_it->_tri_faces[next_face_index++];

                    for (unsigned i = 0; i < 3; ++i) {
                        
                        line >> t._vertices[i];

                        char sep;
                        line.get(sep);
                        // catch the following cases
                        // f v v v
                        // f v/vt v/vt v/vt
                        // f v//vn v//vn v//vn
                        // f v/vt/vn v/vt/vn v/vt/vn
                        switch (sep) {
                            case ' ': {
                                    t._normals[i]    = 0;
                                    t._tex_coords[i] = 0;
                                }
                                break;
                            case '/': {
                                    line.get(sep);
                                    switch (sep) {
                                        case '/': {
                                                line >> t._normals[i];
                                                t._tex_coords[i] = 0;
                                            }
                                            break;
                                        default: {
                                                line.unget();
                                                line >> t._tex_coords[i];

                                                line.get(sep);
                                                switch (sep) {
                                                    case '/': {
                                                            line >> t._normals[i];
                                                        }
                                                        break;
                                                    default: {
                                                            t._normals[i] = 0;
                                                        }
                                                        break;
                                                }

                                             }
                                    }
                                }
                                break;
                        }
                    }
                }
                break;
            case '#':break;
            default:;
        }
    }

    // sanity check
    assert(out_obj._num_vertices   == next_vertex_index);
    assert(out_obj._num_normals    == next_normal_index);
    assert(out_obj._num_tex_coords == next_tex_coord_index);


    obj_file.close();

    return (true);
}

} // namespace scm


#if    SCM_COMPILER     == SCM_COMPILER_MSVC \
    && SCM_COMPILER_VER >= 1400

#pragma warning(pop)                // restore warnings to previous state

#endif //    SCM_COMPILER     == SCM_COMPILER_MSVC
       // && SCM_COMPILER_VER >= 1400



