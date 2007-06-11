
#ifndef VOLUME_H_INCLUDED
#define VOLUME_H_INCLUDED

#include <volume_renderer/volume_renderer_parameters.h>

#include <scm/data/analysis/transfer_function/piecewise_function_1d.h>

struct data_properties
{
    data_properties() : _dimensions(0) {
    }

    math::vec<unsigned, 3>                                          _dimensions;

    scm::data::piecewise_function_1d<unsigned char, float>          _alpha_transfer;
    scm::data::piecewise_function_1d<unsigned char, math::vec3f_t>  _color_transfer;

};

// global data declarations
extern gl::volume_renderer_parameters                      _volrend_params;
extern data_properties                                     _data_properties;


// function prototypes
bool update_color_alpha_table();
bool open_volume_file(const std::string& filename);
bool open_volume();


#endif // VOLUME_H_INCLUDED

