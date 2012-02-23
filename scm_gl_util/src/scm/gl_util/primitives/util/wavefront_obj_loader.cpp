
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "wavefront_obj_loader.h"

#include <scm/core/platform/platform.h>

#if    SCM_COMPILER     == SCM_COMPILER_MSVC \
    && SCM_COMPILER_VER >= 1400

#pragma warning(push)               // preserve warning settings
#pragma warning(disable : 4244)     // disable warning C4244: lexical_cast initializing size_t to int

#endif //    SCM_COMPILER     == SCM_COMPILER_MSVC
       // && SCM_COMPILER_VER >= 1400

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <scm/core/io/tools.h>

#include <scm/gl_util/primitives/util/wavefront_obj_file.h>

namespace scm {
namespace gl {
namespace util {

bool load_material_lib(const std::string& filename, wavefront_model& out_obj)
{
    //std::ifstream   mtl_file;

    std::string file_contents;

    if (!io::read_text_file(filename, file_contents)) {
        return (false);
    }
    std::istringstream mtl_file(file_contents);
    file_contents.clear();

    //mtl_file.open(filename.c_str(), std::ios_base::in);

    //if (!mtl_file) {
    //    return (false);
    //}

    std::string     cur_line;

    wavefront_model::material_container::iterator   cur_material;

    while (std::getline(mtl_file, cur_line)) {
        
        std::istringstream line(cur_line.c_str());

        char line_id;
        line.get(line_id);
        while (   (line_id == ' ')
               || (line_id == '\t')) {
            line.get(line_id);
        }

        switch (line_id) {
            case 'n': {
                    std::string  tag;
                    line.putback(line_id);
                    line >> tag;

                    // load material library
                    if (tag == std::string("newmtl")) {
                        std::string mat_name;
                        line >> mat_name;

                        if (out_obj._materials.find(mat_name) == out_obj._materials.end()) {
                            std::pair<wavefront_model::material_container::iterator, bool> ret;
                            ret = out_obj._materials.insert(wavefront_model::material_container::value_type(mat_name, wavefront_material()));

                            cur_material = ret.first;
                        }
                    }
                }
                break;
            case 'N': {
                    line.get(line_id);
                    switch (line_id) {
                        case 's': {
                                line >> cur_material->second._Ns;
                            }
                            break;
                        case 'i': {
                                line >> cur_material->second._Ni;
                            }
                            break;
                    }
                }
                break;
            case 'T': {
                    line.get(line_id);
                    switch (line_id) {
                        case 'f': {
                                line >> cur_material->second._Tf.x;
                                line >> cur_material->second._Tf.y;
                                line >> cur_material->second._Tf.z;
                            }
                            break;
                    }
                }
                break;
            case 'K': {
                    line.get(line_id);
                    switch (line_id) {
                        case 'a': {
                                line >> cur_material->second._Ka.x;
                                line >> cur_material->second._Ka.y;
                                line >> cur_material->second._Ka.z;
                            }
                            break;
                        case 'd': {
                                line >> cur_material->second._Kd.x;
                                line >> cur_material->second._Kd.y;
                                line >> cur_material->second._Kd.z;
                            }
                            break;
                        case 's': {
                                line >> cur_material->second._Ks.x;
                                line >> cur_material->second._Ks.y;
                                line >> cur_material->second._Ks.z;
                            }
                            break;
                    }
                }
                break;
            case 'd': {
                    line >> cur_material->second._d;
                }
                break;
            case '#':break;
            default:;
        }
    }

    //mtl_file.close();

    return (true);
}

bool open_obj_file(const std::string& filename, wavefront_model& out_obj)
{
    using namespace boost::filesystem;

    std::ifstream   obj_file;

    path                    file_path(filename);
    std::string             file_name       = file_path.filename().string();
    std::string             file_extension  = file_path.extension().string();

    obj_file.open(filename.c_str(), std::ios_base::in);

    if (!obj_file) {
        return (false);
    }

    if (!out_obj._objects.empty()) {
        // clear model structure
        out_obj._objects.clear();
    }

    std::string     cur_line;
    bool            group_definition_started        = false;
    bool            object_definition_started       = false;
    bool            grpmtl_defined                  = false;

    std::string     last_used_material              = "default";

    wavefront_model::object_container::iterator     cur_obj_it = out_obj.add_new_object();;
    wavefront_object::group_container::iterator     cur_grp_it = cur_obj_it->add_new_group();

    // first pass trough the file
    // collect data about file
    while (std::getline(obj_file, cur_line)) {

        std::istringstream line(cur_line.c_str());

        char line_id;
        line.get(line_id);


        switch (line_id) {
            case 'v': {
                    line.get(line_id);
                    switch (line_id) {
                        case ' ': ++out_obj._num_vertices;break;
                        case 'n': ++out_obj._num_normals;break;
                        case 't': ++out_obj._num_tex_coords;break;
                    }
                }
                break;
            case 'f': {
                    ++(cur_grp_it->_num_tri_faces);
                }
                break;
            case 'o': {
                    std::string name;
                    line >> name;

                    if (object_definition_started) {
                        cur_obj_it = out_obj.add_new_object(name);
                        cur_grp_it = cur_obj_it->add_new_group();
                    }
                    else {
                        cur_obj_it->_name = name;
                    }

                    object_definition_started    = true;
                    group_definition_started     = false;
                }
                break;
            case 'm': {
                    std::string  tag;
                    line.putback(line_id);
                    line >> tag;

                    // load material library
                    if (tag == std::string("mtllib")) {

                        std::string matlib_file;
                        line >> matlib_file;

                        path matlib_file_name = file_path.parent_path() / matlib_file;

                        if (!load_material_lib(matlib_file_name.string(), out_obj)) {
                            //out_obj._objects.clear();

                            std::cout << "open_obj_file(): warning: loading materal lib ('"
                                      << matlib_file_name << "')"
                                      << std::endl;
                        }
                    }
                }
                break;
            case 'g': {
                    std::string name;
                    line >> name;

                    if (group_definition_started) {
                        cur_grp_it = cur_obj_it->add_new_group(name);
                    }
                    else {
                        cur_grp_it->_name = name;
                    }

                    group_definition_started     = true;
                }
                break;
            case 'u': {
                    std::string  tag;
                    line.putback(line_id);
                    line >> tag;

                    if (tag == std::string("usemtl")) {
                        std::string mat_name;
                        line >> mat_name;
                        last_used_material = mat_name;

                        if (0 == cur_grp_it->_num_tri_faces) {
                            cur_grp_it->_material_name = mat_name;
                        }
                        else {
                            std::string n = cur_grp_it->_name;
                            cur_grp_it = cur_obj_it->add_new_group(n);
                            cur_grp_it->_material_name = mat_name;
                        }
                        /*if (grpmtl_defined) {
                            cur_grp_it = cur_obj_it->add_new_group();
                        }*/


                        /*cur_grp_it->_material_name  =  mat_name;
                        cur_grp_it->_name           += cur_grp_it->_material_name;*/

                        grpmtl_defined = true;
                    }
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
            out_obj._vertices.reset(new scm::math::vec3f[out_obj._num_vertices]);
        }
        if (out_obj._num_normals != 0) {
            out_obj._normals.reset(new scm::math::vec3f[out_obj._num_normals]);
        }
        if (out_obj._num_tex_coords != 0) {
            out_obj._tex_coords.reset(new scm::math::vec2f[out_obj._num_tex_coords]);
        }

        for (cur_obj_it = out_obj._objects.begin(); cur_obj_it != out_obj._objects.end(); ++cur_obj_it) {
            for (cur_grp_it = cur_obj_it->_groups.begin(); cur_grp_it != cur_obj_it->_groups.end(); ++cur_grp_it) {
                if (cur_grp_it->_num_tri_faces != 0) {
                    cur_grp_it->_tri_faces.reset(new wavefront_object_triangle_face[cur_grp_it->_num_tri_faces]);
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

    group_definition_started    = false;
    object_definition_started   = false;
    grpmtl_defined              = false;

    obj_file.clear();
    obj_file.seekg(0);

    while (std::getline(obj_file, cur_line)) {
        std::istringstream line(cur_line.c_str());

        char line_id;
        line.get(line_id);

        switch (line_id) {
            case 'v': {
                    line.get(line_id);
                    switch (line_id) {
                        case ' ': {
                                scm::math::vec3f&  v = out_obj._vertices[next_vertex_index++];
                                line >> v.x;
                                line >> v.y;
                                line >> v.z;
                            }
                            break;
                        case 'n': {
                                scm::math::vec3f&  n = out_obj._normals[next_normal_index++];
                                line >> n.x;
                                line >> n.y;
                                line >> n.z;
                            }
                            break;
                        case 't': {
                                scm::math::vec2f&  t = out_obj._tex_coords[next_tex_coord_index++];
                                line >> t.x;
                                line >> t.y;
                            }
                            break;
                    }
                }
                break;
            case 'o': {
                    if (object_definition_started) {
                        ++cur_obj_it;
                        cur_grp_it = cur_obj_it->_groups.begin();
                        next_face_index = 0;
                    }
                    object_definition_started   = true;
                    group_definition_started    = false;
                }
                break;
            case 'g': {
                    if (group_definition_started) {
                        ++cur_grp_it;
                        next_face_index = 0;
                    }
                    group_definition_started = true;
                    //grpmtl_defined = false;
                }
                break;
            case 'u': {
                    std::string  tag;
                    line.putback(line_id);
                    line >> tag;

                    if (tag == std::string("usemtl")) 
                    {
                        std::string mat_name;
                        line >> mat_name;
                        last_used_material = mat_name;

                        //if (0 == cur_grp_it->_num_tri_faces) {
                        //    //cur_grp_it->_material_name = mat_name;
                        //}
                        //else {
                        //    //std::string n = cur_grp_it->_name;
                        //    //cur_grp_it = cur_obj_it->add_new_group(n);
                        //    //cur_grp_it->_material_name = mat_name;
                        //    ++cur_grp_it;
                        //    next_face_index = 0;
                        //}

                        if (next_face_index) {
                            ++cur_grp_it;
                            next_face_index = 0;
                            //grpmtl_defined = false;
                        }

                        //grpmtl_defined = true;
                    }
                }
                break;
            case 'f': {
                    wavefront_object_triangle_face& t = cur_grp_it->_tri_faces[next_face_index++];

                    t._material_name = last_used_material;
                    //grpmtl_defined = true;

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

} // namespace util
} // namespace gl
} // namespace scm


#if    SCM_COMPILER     == SCM_COMPILER_MSVC \
    && SCM_COMPILER_VER >= 1400

#pragma warning(pop)                // restore warnings to previous state

#endif //    SCM_COMPILER     == SCM_COMPILER_MSVC
       // && SCM_COMPILER_VER >= 1400



