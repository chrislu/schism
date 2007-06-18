
#define NOMINMAX

#include "volume.h"

#include <iostream>

#include <scm/core/utilities/boost_warning_disable.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/data/volume/scm_vol/scm_vol.h>
#include <scm/data/volume/volume_data_loader.h>
#include <scm/data/volume/volume_data_loader_raw.h>
#include <scm/data/volume/volume_data_loader_vgeo.h>
#include <scm/data/volume/volume_data_loader_svol.h>

#include <scm/data/analysis/transfer_function/build_lookup_table.h>

#if _WIN32
#include <windows.h>
#endif

#include <scm/ogl/utilities/error_checker.h>

#include <scm/core/math/math.h>

gl::volume_renderer_parameters                      _volrend_params = gl::volume_renderer_parameters();
data_properties                                     _data_properties = data_properties();

// implementation
bool open_volume_file(const std::string& filename)
{
    std::cout << "start loading file: " << filename << std::endl;
  
    using namespace boost::filesystem;
   
    boost::scoped_ptr<scm::data::volume_data_loader> vol_loader;    
    path                    file_path(filename, native);
    std::string             file_name       = file_path.leaf();
    std::string             file_extension  = extension(file_path);
    
    boost::algorithm::to_lower(file_extension);

    unsigned                voxel_components;

    if (file_extension == ".raw") {
        vol_loader.reset(new scm::data::volume_data_loader_raw());
    }
    else if (file_extension == ".vol") {
        vol_loader.reset(new scm::data::volume_data_loader_vgeo());
    }
    else if (file_extension == ".svol") {
        vol_loader.reset(new scm::data::volume_data_loader_svol());
    }
    else {
        return (false);
    }

    if (!vol_loader->open_file(filename)) {
        return (false);
    }

    math::vec<unsigned, 3> data_dimensions;

    data_dimensions = vol_loader->get_volume_descriptor()._data_dimensions;

    // get max gl 3d tex dim
    int gl_max_3dtex_dim;
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, & gl_max_3dtex_dim);

    data_dimensions.x = math::clamp(data_dimensions.x, 0u, (unsigned int)gl_max_3dtex_dim);
    data_dimensions.y = math::clamp(data_dimensions.y, 0u, (unsigned int)gl_max_3dtex_dim);
    data_dimensions.z = math::clamp(data_dimensions.z, 0u, (unsigned int)gl_max_3dtex_dim);

    unsigned max_dim = math::max(data_dimensions.x, math::max(data_dimensions.y, data_dimensions.z));

    _volrend_params._aspect.x = (float)data_dimensions.x/(float)max_dim;
    _volrend_params._aspect.y = (float)data_dimensions.y/(float)max_dim;
    _volrend_params._aspect.z = (float)data_dimensions.z/(float)max_dim;

    //_volrend_params._aspect = _volrend_params._aspect * vol_loader->get_volume_descriptor()._volume_aspect;

    scm::data::regular_grid_data_3d<unsigned char> data;

    if (!vol_loader->read_sub_volume(math::vec<unsigned, 3>(0, 0, 0), data_dimensions, data)) {
        return (false);
    }
    _data_properties._vol_desc  = vol_loader->get_volume_descriptor();
    _data_properties._vol_scale = math::vec3f_t(float(max_dim));

    vol_loader->close_file();

    _data_properties._dimensions = data_dimensions;
    voxel_components = 1;

    // analysis! -> histogram etc...

    //std::cout << "start histogram calculation" << std::endl;
    //if (!scm::histogram_1d_calculator<unsigned char>::calculate(_data_properties._histogram, data)) {
    //    std::cout << "error during histogram calculation" << std::endl;
    //    return (false);
    //}
    //std::cout << "end histogram calculation" << std::endl;

    // load opengl texture
    GLenum internal_format;
    GLenum source_format;

    switch (voxel_components) {
        case 1:internal_format = GL_LUMINANCE; source_format = GL_LUMINANCE; break;
        case 2:internal_format = GL_LUMINANCE_ALPHA; source_format = GL_LUMINANCE_ALPHA; break;
        case 3:internal_format = GL_RGB; source_format = GL_RGB; break;
        case 4:internal_format = GL_RGBA; source_format = GL_RGBA; break;
        default: return (false);
    }

    if (!_volrend_params._volume_texture.tex_image( 0,
                             internal_format,
                             _data_properties._dimensions.x,
                             _data_properties._dimensions.y,
                             _data_properties._dimensions.z,
                             source_format,
                             GL_UNSIGNED_BYTE,
                             data.get_data().get())) {

        std::cout << "end texture upload - FAILED reason: ";
        scm::gl::error_checker ech;
        std::cout << ech.get_error_string(_volrend_params._volume_texture.get_last_error()) << std::endl;
        return (false);
    }

    std::cout << "end texture upload" << std::endl;

    _volrend_params._volume_texture.bind();
    _volrend_params._volume_texture.tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR); // GL_NEAREST); // 
    _volrend_params._volume_texture.tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR); // GL_NEAREST); // 
    _volrend_params._volume_texture.tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    _volrend_params._volume_texture.tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    _volrend_params._volume_texture.tex_parameteri(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    _volrend_params._volume_texture.unbind();

    
//#define NOT_VRGEO
    // reset transfer functions
    _data_properties._alpha_transfer.clear();

#ifdef NOT_VRGEO
    _data_properties._alpha_transfer.add_point(0,   0.0f);
    //_data_properties._alpha_transfer.add_point(90, 0.0f);
    //_data_properties._alpha_transfer.add_point(100, 1.0f);
    _data_properties._alpha_transfer.add_point(255, 1.0f);
#else
    _data_properties._alpha_transfer.add_point(0,   1.0f);
    //_data_properties._alpha_transfer.add_point(75,  1.00f);
    //_data_properties._alpha_transfer.add_point(80,  0.00f);
    //_data_properties._alpha_transfer.add_point(128, 0.0f);
    //_data_properties._alpha_transfer.add_point(170, 0.00f);
    //_data_properties._alpha_transfer.add_point(175, 1.00f);
    _data_properties._alpha_transfer.add_point(100,  0.00f);
    _data_properties._alpha_transfer.add_point(128, 0.0f);
    _data_properties._alpha_transfer.add_point(160, 0.00f);
    _data_properties._alpha_transfer.add_point(255, 1.0f);
#endif

    _data_properties._color_transfer.clear();
#ifdef NOT_VRGEO
    _data_properties._color_transfer.add_point(0,   math::vec3f_t(0.0f, 0.0f, 0.0f));
    //_data_properties._color_transfer.add_point(90,  math::vec3f_t(0.9f, 0.8f, 0.6f));
    _data_properties._color_transfer.add_point(255, math::vec3f_t(1.0f, 1.0f, 1.0f));
#else
    _data_properties._color_transfer.add_point(0,   math::vec3f_t(0.0f, 0.0f, 1.0f));
    _data_properties._color_transfer.add_point(128, math::vec3f_t(1.0f, 1.0f, 1.0f));
    _data_properties._color_transfer.add_point(255, math::vec3f_t(1.0f, 0.0f, 0.0f));
#endif

    update_color_alpha_table();

    _volrend_params._step_size      = 512;
    _volrend_params._voxel_size.x   = 1.0f / float(_data_properties._dimensions.x);
    _volrend_params._voxel_size.y   = 1.0f / float(_data_properties._dimensions.y);
    _volrend_params._voxel_size.z   = 1.0f / float(_data_properties._dimensions.z);

    std::cout << "end loading file: " << filename << std::endl;

    return (true);
}

bool open_unc_volume_file(const std::string& filename)
{
    std::cout << "start loading file: " << filename << std::endl;
  
    using namespace boost::filesystem;
   
    boost::scoped_ptr<scm::data::volume_data_loader> vol_loader;    
    path                    file_path(filename, native);
    std::string             file_name       = file_path.leaf();
    std::string             file_extension  = extension(file_path);
    
    boost::algorithm::to_lower(file_extension);

    unsigned                voxel_components;

    if (file_extension == ".raw") {
        vol_loader.reset(new scm::data::volume_data_loader_raw());
    }
    else if (file_extension == ".vol") {
        vol_loader.reset(new scm::data::volume_data_loader_vgeo());
    }
    else if (file_extension == ".svol") {
        vol_loader.reset(new scm::data::volume_data_loader_svol());
    }
    else {
        return (false);
    }

    if (!vol_loader->open_file(filename)) {
        return (false);
    }

    math::vec<unsigned, 3> data_dimensions;

    data_dimensions = vol_loader->get_volume_descriptor()._data_dimensions;

    // get max gl 3d tex dim
    int gl_max_3dtex_dim;
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, & gl_max_3dtex_dim);

    data_dimensions.x = math::clamp(data_dimensions.x, 0u, (unsigned int)gl_max_3dtex_dim);
    data_dimensions.y = math::clamp(data_dimensions.y, 0u, (unsigned int)gl_max_3dtex_dim);
    data_dimensions.z = math::clamp(data_dimensions.z, 0u, (unsigned int)gl_max_3dtex_dim);

    //_volrend_params._aspect = _volrend_params._aspect * vol_loader->get_volume_descriptor()._volume_aspect;

    scm::data::regular_grid_data_3d<unsigned char> data;

    if (!vol_loader->read_sub_volume(math::vec<unsigned, 3>(0, 0, 0), data_dimensions, data)) {
        return (false);
    }

    vol_loader->close_file();

    _data_properties._dimensions = data_dimensions;
    voxel_components = 1;

    // analysis! -> histogram etc...

    //std::cout << "start histogram calculation" << std::endl;
    //if (!scm::histogram_1d_calculator<unsigned char>::calculate(_data_properties._histogram, data)) {
    //    std::cout << "error during histogram calculation" << std::endl;
    //    return (false);
    //}
    //std::cout << "end histogram calculation" << std::endl;

    // load opengl texture
    GLenum internal_format;
    GLenum source_format;

    switch (voxel_components) {
        case 1:internal_format = GL_LUMINANCE; source_format = GL_LUMINANCE; break;
        case 2:internal_format = GL_LUMINANCE_ALPHA; source_format = GL_LUMINANCE_ALPHA; break;
        case 3:internal_format = GL_RGB; source_format = GL_RGB; break;
        case 4:internal_format = GL_RGBA; source_format = GL_RGBA; break;
        default: return (false);
    }

    if (!_volrend_params._uncertainty_volume_texture.tex_image( 0,
                             internal_format,
                             _data_properties._dimensions.x,
                             _data_properties._dimensions.y,
                             _data_properties._dimensions.z,
                             source_format,
                             GL_UNSIGNED_BYTE,
                             data.get_data().get())) {

        std::cout << "end texture upload - FAILED reason: ";
        scm::gl::error_checker ech;
        std::cout << ech.get_error_string(_volrend_params._uncertainty_volume_texture.get_last_error()) << std::endl;
        return (false);
    }

    std::cout << "end texture upload" << std::endl;

    _volrend_params._uncertainty_volume_texture.bind();
    _volrend_params._uncertainty_volume_texture.tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR); // GL_NEAREST); // 
    _volrend_params._uncertainty_volume_texture.tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR); // GL_NEAREST); // 
    _volrend_params._uncertainty_volume_texture.tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    _volrend_params._uncertainty_volume_texture.tex_parameteri(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    _volrend_params._uncertainty_volume_texture.tex_parameteri(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    _volrend_params._uncertainty_volume_texture.unbind();

    
    std::cout << "end loading file: " << filename << std::endl;

    return (true);
}

bool update_color_alpha_table()
{
    boost::scoped_array<math::vec3f_t>  color_lut;
    boost::scoped_array<float>          alpha_lut;

    unsigned                            lut_texture_size = 512;

    if (color_lut.get() == 0) {
        try {
            color_lut.reset(new math::vec3f_t[lut_texture_size]);
        }
        catch (std::bad_alloc&) {
            std::cout << "error allocationg color lookuptable" << std::endl;
            color_lut.reset();
            return (false);
        }
    }
    if (alpha_lut.get() == 0) {
        try {
            alpha_lut.reset(new float[lut_texture_size]);
        }
        catch (std::bad_alloc&) {
            std::cout << "error allocationg alpha lookuptable" << std::endl;
            alpha_lut.reset();
            return (false);
        }
    }

    if (!scm::data::build_lookup_table(color_lut, _data_properties._color_transfer, lut_texture_size)) {
        std::cout << "error during lookuptable generation" << std::endl;
        return (false);
    }


    if (!scm::data::build_lookup_table(alpha_lut, _data_properties._alpha_transfer, lut_texture_size)) {
        std::cout << "error during lookuptable generation" << std::endl;
        return (false);
    }
    boost::scoped_array<float> combined_lut;

    try {
        combined_lut.reset(new float[lut_texture_size * 4]);
    }
    catch (std::bad_alloc&) {
        std::cout << "critical error" << std::endl;
        combined_lut.reset();
        return (false);
    }

    for (unsigned i = 0; i < lut_texture_size; ++i) {
        combined_lut[i*4   ] = color_lut[i].x;
        combined_lut[i*4 +1] = color_lut[i].y;
        combined_lut[i*4 +2] = color_lut[i].z;
        combined_lut[i*4 +3] = alpha_lut[i];
    }
    
    std::cout << "end lookuptable calculation" << std::endl;

    // load opengl texture
    // load opengl texture
    unsigned pixel_components = 4;
    GLenum internal_format;
    GLenum source_format;

    switch (pixel_components) {
        case 1:internal_format = GL_LUMINANCE16F_ARB; source_format = GL_LUMINANCE; break;
        case 2:internal_format = GL_LUMINANCE_ALPHA16F_ARB; source_format = GL_LUMINANCE_ALPHA; break;
        case 3:internal_format = GL_RGB16F_ARB; source_format = GL_RGB; break;
        case 4:internal_format = GL_RGBA16F_ARB; source_format = GL_RGBA; break;
        default: return (false);
    }

    if (!_volrend_params._color_alpha_texture.tex_image( 0,
                             internal_format,
                             lut_texture_size,
                             source_format,
                             GL_FLOAT,
                             combined_lut.get())) {

        std::cout << "end texture upload - FAILED reason: ";
        scm::gl::error_checker ech;
        std::cout << ech.get_error_string(_volrend_params._color_alpha_texture.get_last_error()) << std::endl;
        return (false);
    }

    _volrend_params._color_alpha_texture.bind();
    _volrend_params._color_alpha_texture.tex_parameteri(GL_TEXTURE_MIN_FILTER, GL_LINEAR); // GL_NEAREST); // 
    _volrend_params._color_alpha_texture.tex_parameteri(GL_TEXTURE_MAG_FILTER, GL_LINEAR); // GL_NEAREST); // 
    _volrend_params._color_alpha_texture.tex_parameteri(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    _volrend_params._color_alpha_texture.unbind();

    return (true);
}

bool open_volume()
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
    ofn.lpstrFilter = "all files (*.*)\0*.*\0schism volume files (*.svol)\0*.svol\0raw volume files (*.raw)\0*.raw\0voxel geo volume files (*.vol)\0*.vol\0";
    ofn.nFilterIndex = 2;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    // Display the Open dialog box. 

    if (GetOpenFileName(&ofn)==TRUE) {
        if (!open_volume_file(std::string(ofn.lpstrFile)))
            return (false);
    }
    else {
        return (false);
    }
#else
    if (!open_volume_file(std::string("/mnt/data/_devel/data/vrgeo/wfarm_200_w512_h439_d512_c1_b8.raw"))) {
        return (false);
    } 
#endif
    return (true);
}

bool open_unc_volume()
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
    ofn.lpstrFilter = "all files (*.*)\0*.*\0schism volume files (*.svol)\0*.svol\0raw volume files (*.raw)\0*.raw\0voxel geo volume files (*.vol)\0*.vol\0";
    ofn.nFilterIndex = 2;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

    // Display the Open dialog box. 

    if (GetOpenFileName(&ofn)==TRUE) {
        if (!open_unc_volume_file(std::string(ofn.lpstrFile)))
            return (false);
    }
    else {
        return (false);
    }
#else
    if (!open_volume_file(std::string("/mnt/data/_devel/data/vrgeo/wfarm_200_w512_h439_d512_c1_b8.raw"))) {
        return (false);
    } 
#endif
    return (true);
}
