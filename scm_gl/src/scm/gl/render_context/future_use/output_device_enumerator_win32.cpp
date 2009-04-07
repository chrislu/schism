
#include "output_device_enumerator_win32.h"

#include <windows.h>

#include <ogl/render_context/wgl_extensions.h>

#include <set>
#include <boost/lexical_cast.hpp>

namespace gl
{
    output_device_enumerator_win32::output_device_enumerator_win32()
    {
    }

    output_device_enumerator_win32::~output_device_enumerator_win32()
    {
    }

    bool output_device_enumerator_win32::enumerate_devices(std::vector<gl::output_device_descriptor>& devices)
    {
        _issues.push_back(std::string("output_device_enumerator_win32::enumerate_devices(): {"));
        unsigned num_output_devices = (unsigned)GetSystemMetrics(SM_CMONITORS);

        if (num_output_devices < 1) {
            _issues.push_back(std::string("Error: GetSystemMetrics returned < 1 monitors"));
            return (false);
        }

        for (unsigned i = 0; i < num_output_devices; i++) {
            gl::output_device_descriptor dev_desc;
            _issues.push_back(std::string("\tenumerating device: ") + boost::lexical_cast<std::string>(i));

            if (!enumerate_device(dev_desc, i)) {
                continue;
            }

            devices.push_back(dev_desc);
        }
        
        _issues.push_back(std::string("}"));

        return (true);
    }

    bool output_device_enumerator_win32::enumerate_device(gl::output_device_descriptor& desc, const std::string& name)
    {
        _issues.push_back(std::string("output_device_enumerator_win32::enumerate_device(): {"));

        DISPLAY_DEVICE               disp_device;

        disp_device.cb = sizeof(DISPLAY_DEVICE);

        unsigned            dev_index = 0;
        bool                found_dev = false;

        while (!found_dev && EnumDisplayDevices(NULL, dev_index, &disp_device, 0))  {
            if (name == std::string(disp_device.DeviceName)) {
                found_dev = true;
            }
            else {
                dev_index++;
            }
        }

        if (found_dev) {
            return (this->enumerate_device(desc, dev_index));
        }
        else {
            _issues.push_back(std::string("\tError: unable to find device: ") + name);
            return (false);
        }
    }

    bool output_device_enumerator_win32::enumerate_device(gl::output_device_descriptor& desc, unsigned device)
    {
        _issues.push_back(std::string("output_device_enumerator_win32::enumerate_device(): {"));
        //unsigned num_output_devices = (unsigned)GetSystemMetrics(SM_CMONITORS);

        //if (num_output_devices < device) {
        //    _issues.push_back(std::string("Error: GetSystemMetrics returned < device_num monitors"));
        //    return (false);
        //}

        DISPLAY_DEVICE               disp_device;
        DEVMODE                      dev_mode;

        disp_device.cb = sizeof(DISPLAY_DEVICE);

        if (!EnumDisplayDevices(NULL, device, &disp_device, 0)) {
            _issues.push_back(std::string("Error: EnumDisplayDevices returned false for device num ") + boost::lexical_cast<std::string>(device));
            return (false);
        }

        desc._device_name   = std::string(disp_device.DeviceName);
        desc._device_string = std::string(disp_device.DeviceString);
        desc._device_id     = std::string(disp_device.DeviceID);


        int dev_mode_num = 0;
        while (EnumDisplaySettings((const char*)disp_device.DeviceName, dev_mode_num, &dev_mode)) {
            gl::output_device_descriptor::device_mode_attribs out_dev_mode;
            
            out_dev_mode._width             = dev_mode.dmPelsWidth;
            out_dev_mode._height            = dev_mode.dmPelsHeight;
            out_dev_mode._refresh_rate      = dev_mode.dmDisplayFrequency;

            desc._device_mode_infos[dev_mode.dmBitsPerPel]._avail_device_modes.push_back(out_dev_mode);

            dev_mode_num++;
        }

        // enumerate pixel formats
        if (!wgl::is_initialized()) {
            if (!wgl::initialize_wgl()) {
                return (false);
            }
        }

        int*              pixel_fmts = NULL;
        unsigned int      num_pixel_fmts;
        HDC               dev_context = NULL;

        dev_context = CreateDC(disp_device.DeviceName, disp_device.DeviceName, 0, NULL);
        //dev_context = CreateDC(TEXT("DISPLAY"), disp_device.DeviceName, 0, NULL);

        int query_num_attrib[] = { WGL_NUMBER_PIXEL_FORMATS_ARB };
        int query_num_values[2];

        gl::output_device_descriptor::device_mode_info_map_t::iterator out_dev_mode_it;
        for (out_dev_mode_it = desc._device_mode_infos.begin();
             out_dev_mode_it != desc._device_mode_infos.end();
             out_dev_mode_it++) {

            // query max number of pixel formats
            if (!wgl::wglGetPixelFormatAttribiv(dev_context, 0, 0, 1, query_num_attrib, query_num_values)) {
                _issues.push_back(std::string("Warning: wglGetPixelFormatAttribiv returned false for device num ") 
                                  + boost::lexical_cast<std::string>(device)
                                  + std::string("\n\twhen querying max formats for display mode at bpp: "
                                  + boost::lexical_cast<std::string>(out_dev_mode_it->first)));
                continue;
            }
            num_pixel_fmts = query_num_values[0];
            //pixel_fmts = new int[num_pixel_fmts];

            if (num_pixel_fmts < 1) {
                _issues.push_back(std::string("Warning: wglGetPixelFormatAttribiv returned 0 pixelformats for devide num") 
                                  + boost::lexical_cast<std::string>(device)
                                  + std::string("\n\twhen querying max formats for display mode at bpp: "
                                  + boost::lexical_cast<std::string>(out_dev_mode_it->first)));
                continue;
            }

            int query_fmt_attrib[] = { WGL_COLOR_BITS_ARB,
                                       WGL_DEPTH_BITS_ARB,
                                       WGL_ALPHA_BITS_ARB,
                                       WGL_STENCIL_BITS_ARB,
                                       WGL_AUX_BUFFERS_ARB,
                                       WGL_SAMPLES_ARB,
                                       WGL_DOUBLE_BUFFER_ARB,
                                       WGL_STEREO_ARB,

                                       //WGL_DRAW_TO_WINDOW_ARB,
                                       WGL_ACCELERATION_ARB,
                                       WGL_SUPPORT_OPENGL_ARB,
                                       WGL_PIXEL_TYPE_ARB
                                     };
            int query_fmt_values[12];

            for (unsigned f = 1; f <= num_pixel_fmts /*num_supported_pixel_fmts*/; f++) {
                gl::output_device_descriptor::pixel_format_attribs context_attribs;

                if (!wgl::wglGetPixelFormatAttribiv(dev_context, f/*pixel_fmts[f]*/, 0, 11, query_fmt_attrib, query_fmt_values)) {
                _issues.push_back(std::string("Warning: wglGetPixelFormatAttribiv returned false for device num ") 
                                  + boost::lexical_cast<std::string>(device)
                                  + std::string("\n\twhen querying format attribs for pixel format:")
                                  + boost::lexical_cast<std::string>(f)
                                  + std::string("\n\tat display mode bpp: ")
                                  + boost::lexical_cast<std::string>(out_dev_mode_it->first));
                    continue;
                }

                if (   query_fmt_values[0]  == out_dev_mode_it->first
                    && query_fmt_values[8]  == WGL_FULL_ACCELERATION_ARB
                    && query_fmt_values[9]  == GL_TRUE
                    && query_fmt_values[10] == WGL_TYPE_RGBA_ARB) {
                        
                    out_dev_mode_it->second._avail_pixel_formats._avail_depth_bits.insert(   query_fmt_values[1]);
                    out_dev_mode_it->second._avail_pixel_formats._avail_alpha_bits.insert(   query_fmt_values[2]);
                    out_dev_mode_it->second._avail_pixel_formats._avail_stencil_bits.insert( query_fmt_values[3]);
                    out_dev_mode_it->second._avail_pixel_formats._avail_aux_buffers.insert(  query_fmt_values[4]);
                    out_dev_mode_it->second._avail_pixel_formats._avail_samples.insert(      query_fmt_values[5]);
                    out_dev_mode_it->second._avail_pixel_formats._avail_doublebuffer.insert( query_fmt_values[6] == GL_TRUE ? true : false);
                    out_dev_mode_it->second._avail_pixel_formats._avail_stereo.insert(       query_fmt_values[7] == GL_TRUE ? true : false);
                }
            }
        }


        DeleteDC(dev_context);
       
        _issues.push_back(std::string("}"));

        return (true);
    }
} // namespace gl
