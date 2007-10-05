
#include "geometry.h"

#include <iostream>
#include <fstream>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/assign/std/vector.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/data/geometry/wavefront_obj/obj_file.h>
#include <scm/data/geometry/wavefront_obj/obj_loader.h>
#include <scm/data/geometry/wavefront_obj/obj_to_vertex_array.h>
#include <scm/core/utilities/foreach.h>

#if _WIN32
#include <windows.h>
#endif

std::vector<geometry>        _geometries = std::vector<geometry>();

bool open_geometry_file(const std::string& filename)
{
    std::cout << "start loading file: " << filename << std::endl;
  
    using namespace boost::filesystem;
   
    path                    file_path(filename, native);
    std::string             file_name       = file_path.leaf();
    std::string             file_extension  = extension(file_path);
    
    boost::algorithm::to_lower(file_extension);

    if (file_extension != ".sgeom") {
        std::cout << "only sgeom files supported by this application!" << std::endl;
        return (false);
    }

    std::ifstream sgeom_file;

    sgeom_file.open(filename.c_str(), std::ios::in);
    if (!sgeom_file) {
        std::cout << "unable to open sgeom file: " << filename << std::endl;
        return (false);
    }

    scm::data::geometry_descriptor geom_desc;

    sgeom_file >> geom_desc;

    if (!sgeom_file) {
        std::cout << "unable to read information from sgeom file: " << filename << std::endl;
        sgeom_file.close();
        return (false);
    }

    sgeom_file.close();

    std::string obj_file_name = (file_path.branch_path() / geom_desc._sobj_file).file_string();

    scm::data::wavefront_model obj_f;

    if (!scm::data::open_obj_file(obj_file_name, obj_f)) {
        std::cout << "failed to parse obj file: " << obj_file_name << std::endl;
        return (false);
    }
    else {
        std::cout << "done parsing obj file: " << obj_file_name << std::endl;
    }


    scm::data::vertexbuffer_data        obj_vbuf;


    if (!scm::data::generate_vertex_buffer(obj_f,
                                           obj_vbuf)) {
        std::cout << "failed to generate vertex buffer for: " << obj_file_name << std::endl;
        return (false);
    }

    using namespace scm::gl;
    using namespace boost::assign;

    geometry    new_geometry;

    new_geometry._vbo.reset(new vertexbuffer());

    vertex_layout::element_container vert_elem;
    vert_elem += vertex_element(vertex_element::position,
                                vertex_element::dt_vec3f);
    vert_elem += vertex_element(vertex_element::normal,
                                vertex_element::dt_vec3f);

    vertex_layout      vert_lo(vert_elem);

    if (! new_geometry._vbo->vertex_data(obj_vbuf._vert_array_count,
                                         vert_lo,
                                         obj_vbuf._vert_array.get())) {
        std::cout << "error uploading vertex data" << std::endl;
        return (false);
    }

    element_layout     elem_lo(element_layout::triangles,
                               element_layout::dt_uint);

    std::size_t face_count = 0;
    for (unsigned i = 0; i < obj_vbuf._index_arrays.size(); ++i) {
        new_geometry._indices.push_back(geometry::index_buffer_container::value_type());
        new_geometry._indices.back().reset(new indexbuffer());


        if (!new_geometry._indices.back()->element_data(obj_vbuf._index_array_counts[i],
                                           elem_lo,
                                           obj_vbuf._index_arrays[i].get())) {
            std::cout << "error uploading element data of indexbuffer number: " << i << std::endl;
            return (false);
        }

        new_geometry._materials.push_back(obj_vbuf._materials[i]);

        face_count += obj_vbuf._index_array_counts[i] / 3;
    }

    std::cout << "num faces: " << face_count << std::endl;


    new_geometry._desc          = geom_desc;
    new_geometry._face_count    = face_count;

    _geometries.push_back(new_geometry);

    return (true);
}

bool open_geometry()
{
#if _WIN32
    OPENFILENAME ofn;       // common dialog box structure
    char szFile[MAX_PATH];  // buffer for file name

    // Initialize OPENFILENAME
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = szFile;
    //
    // Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
    // use the contents of szFile to initialize itself.
    //
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "all files (*.*)\0*.*\0sgeom meta geometry files (*.sgeom)\0*.sgeom\0";
    ofn.nFilterIndex = 2;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    // Display the Open dialog box. 

    if (GetOpenFileName(&ofn)==TRUE) {
        if (!open_geometry_file(std::string(ofn.lpstrFile)))
            return (false);
    }
    else {
        return (false);
    }
#else
    if (!open_geometry_file(std::string("..."))) {
        return (false);
    } 
#endif
    return (true);
}

