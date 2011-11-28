
#ifndef RENDER_DEVICE_DESCRIPTOR_H_INCLUDED
#define RENDER_DEVICE_DESCRIPTOR_H_INCLUDED

#include <defines_clr.h>

#include <ogl/render_context/context_format.h>
#include <ogl/render_context/output_device_mode.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace gl
{
    CLR_PUBLIC struct output_device_descriptor
    {
    private:
        // types
        struct device_mode_attribs
        {
            unsigned    _width;
            unsigned    _height;
            unsigned    _refresh_rate;
        };

        struct pixel_format_attribs
        {
            std::set<unsigned>   _avail_depth_bits;
            std::set<unsigned>   _avail_alpha_bits;
            std::set<unsigned>   _avail_stencil_bits;
            std::set<unsigned>   _avail_aux_buffers;
            std::set<unsigned>   _avail_samples;
            std::set<unsigned>   _avail_doublebuffer;
            std::set<unsigned>   _avail_stereo;
        };

        struct device_output_modes
        {
            std::vector<device_mode_attribs> _avail_device_modes;
            pixel_format_attribs             _avail_pixel_formats;
        };

    public:
        typedef std::vector<device_mode_attribs>        device_mode_vec_t;
        typedef std::map<unsigned, device_output_modes> device_mode_info_map_t;
        typedef std::set<unsigned>                      pixel_format_attrib_set_t;

        const device_mode_info_map_t& get_device_mode_infos() const { return (_device_mode_infos); }

        const std::string&          get_device_name() const { return (this->_device_name); }
        const std::string&          get_device_string() const { return (this->_device_string); }
        const std::string&          get_device_id() const { return (this->_device_id); }

    private:
        std::string                 _device_name;
        std::string                 _device_string;
        std::string                 _device_id;

        device_mode_info_map_t      _device_mode_infos;

        friend class output_device_enumerator;
        friend class output_device_enumerator_win32;


    };

    extern output_device_descriptor NULL_DEVICE_DESCRIPTOR;
} // namespace gl

#endif // RENDER_DEVICE_DESCRIPTOR_H_INCLUDED
