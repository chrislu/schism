
#include "volume_renderer_parameters.h"

namespace gl
{
    volume_renderer_parameters::volume_renderer_parameters()
        : _step_size(32.0f),
          _point_of_interest(0.5f),
          _extend(1.0f),
          _aspect(1.0f),
          _cross_plane_positions(0.2f),
          _cp_enabled(false)
    {
    }
    
    volume_renderer_parameters::volume_renderer_parameters(const gl::volume_renderer_parameters& prop)
        : _step_size(prop._step_size),
          _point_of_interest(prop._point_of_interest),
          _extend(prop._extend),
          _aspect(prop._aspect),
          _cross_plane_positions(prop._cross_plane_positions),
          _volume_texture(prop._volume_texture),
          _color_alpha_texture(prop._color_alpha_texture),
          _cp_enabled(prop._cp_enabled)
    {
    }

    volume_renderer_parameters::~volume_renderer_parameters()
    {
    }

    const volume_renderer_parameters& volume_renderer_parameters::operator=(const gl::volume_renderer_parameters& rhs)
    {
        _cp_enabled                 = rhs._cp_enabled;
        _step_size                  = rhs._step_size;
        _point_of_interest          = rhs._point_of_interest;
        _extend                     = rhs._extend;
        _aspect                     = rhs._aspect;
        _cross_plane_positions      = rhs._cross_plane_positions;
        _volume_texture             = rhs._volume_texture;
        _color_alpha_texture        = rhs._color_alpha_texture;

        return (*this);
    }



} // namespace gl
