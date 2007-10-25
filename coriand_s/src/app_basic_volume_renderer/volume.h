
#ifndef VOLUME_H_INCLUDED
#define VOLUME_H_INCLUDED

#include <scm/core/math/math.h>

#include <scm/data/volume/scm_vol/scm_vol.h>

#include <volume_renderer/volume_renderer_parameters.h>

#include <scm/data/analysis/transfer_function/piecewise_function_1d.h>

struct data_properties
{
    data_properties() : _dimensions(0) {
    }

    math::vec<unsigned, 3>                                          _dimensions;

    scm::data::piecewise_function_1d<unsigned char, float>          _alpha_transfer;
    scm::data::piecewise_function_1d<unsigned char, math::vec3f_t>  _color_transfer;

    scm::data::volume_descriptor                                    _vol_desc;
    math::vec3f_t                                                   _vol_scale;

};

// global data declarations
extern gl::volume_renderer_parameters                      _volrend_params;
extern data_properties                                     _data_properties;


// function prototypes
bool update_color_alpha_table();
bool open_unc_volume_file(const std::string& filename);
bool open_unc_volume();
bool open_volume_file(const std::string& filename);
bool open_volume();


#endif // VOLUME_H_INCLUDED

