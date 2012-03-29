
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include <iostream>
#include <limits>
#include <conio.h>

#include <boost/lexical_cast.hpp>

#include <Windows.h>

#include "wgl.h"

#include <scm/core.h>

#include <scm/gl_core/window_management/wm_fwd.h>
#include <scm/gl_core/window_management/context.h>
#include <scm/gl_core/window_management/display.h>
#include <scm/gl_core/window_management/window.h>

bool output_device_info()
{
    unsigned num_output_devices = (unsigned)GetSystemMetrics(SM_CMONITORS);

    if (num_output_devices < 1) {
        std::cerr << "Error: GetSystemMetrics returned < 1 monitors" << std::endl;
        return (false);
    }

    for (unsigned i = 0; i < num_output_devices; i++) {
        std::cout << "enumerating device: " << boost::lexical_cast<std::string>(i) << std::endl;

        std::cout << "output_device_enumerator_win32::enumerate_device(): {" << std::endl;
        //unsigned num_output_devices = (unsigned)GetSystemMetrics(SM_CMONITORS);

        //if (num_output_devices < device) {
        //    std::cerr << "Error: GetSystemMetrics returned < device_num monitors"));
        //    return (false);
        //}

        DISPLAY_DEVICE               disp_device;
        DISPLAY_DEVICE               disp_monitor;
        DEVMODE                      dev_mode;

        disp_device.cb = sizeof(DISPLAY_DEVICE);
        disp_monitor.cb = sizeof(DISPLAY_DEVICE);

        if (!EnumDisplayDevices(NULL, i, &disp_device, 0)) {
            std::cerr << "Error: EnumDisplayDevices returned false for device num " + boost::lexical_cast<std::string>(i);
            return (false);
        }

        std::cout << "device_name   = " << std::string(disp_device.DeviceName) << std::endl;
        std::cout << "device_string = " << std::string(disp_device.DeviceString) << std::endl;
        std::cout << "device_key    = " << std::string(disp_device.DeviceKey) << std::endl;
        std::cout << "device_id     = " << std::string(disp_device.DeviceID) << std::endl;
#if 0
        DWORD mon = 0;
		while (EnumDisplayDevices(disp_device.DeviceName, mon, &disp_monitor, 0)) { //EDD_GET_DEVICE_INTERFACE_NAME)) {
            std::cout << mon << " monitor_name   = " << std::string(disp_monitor.DeviceName) << std::endl;
            std::cout << mon << " monitor_string = " << std::string(disp_monitor.DeviceString) << std::endl;
            std::cout << mon << " monitor_key    = " << std::string(disp_monitor.DeviceKey) << std::endl;
            std::cout << mon << " monitor_id     = " << std::string(disp_monitor.DeviceID) << std::endl;
			std::cout << mon << " monitor_state  = " << std::hex << (disp_monitor.StateFlags) << std::dec << std::endl;

			HDC               mon_context = NULL;
			//mon_context = CreateDC(disp_device.DeviceName, disp_monitor.DeviceName, 0, NULL);
			mon_context = CreateIC(disp_device.DeviceName, disp_monitor.DeviceString, 0, NULL);

			if (!mon_context) {
				std::cerr << "error creating mon_context" << std::endl;
				char* error_msg;

				if (0 == FormatMessage(       FORMAT_MESSAGE_IGNORE_INSERTS
											| FORMAT_MESSAGE_FROM_SYSTEM
											| FORMAT_MESSAGE_ALLOCATE_BUFFER,
											0,
											GetLastError(),
											MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
											(LPTSTR)&error_msg,
											0,
											0)) {
					std::cerr << "FormatMessage failed (" + boost::lexical_cast<std::string>(GetLastError()) << std::endl;
				}
				std::string msg(error_msg);
				LocalFree(error_msg);

				std::cerr << msg << std::endl;
				return (false);
			}

			int disp_width  = ::GetDeviceCaps(mon_context, HORZRES);//GetSystemMetrics(SM_CXVIRTUALSCREEN);//
			int disp_height = ::GetDeviceCaps(mon_context, VERTRES);

			std::cout << mon << " monitor_width    = " << disp_width << std::endl;
			std::cout << mon << " monitor_height   = " << disp_height << std::endl;

			DeleteDC(mon_context);
            ++mon;
        }
#endif


        if (EnumDisplaySettings((const char*)disp_device.DeviceName, ENUM_CURRENT_SETTINGS, &dev_mode)) {

            std::cout << "dev_name = " << dev_mode.dmDeviceName
                      << ",\tdev_mode._width = " << dev_mode.dmPelsWidth
                      << ",\t dev_mode._height = " << dev_mode.dmPelsHeight
                      << ",\t dev_mode._refresh_rate = " << dev_mode.dmDisplayFrequency
                      << ",\t dev_mode._bpp = " << dev_mode.dmBitsPerPel
                      << ",\t dev_pos.x = " << dev_mode.dmPosition.x
                      << ",\t dev_pos.y = " << dev_mode.dmPosition.y
                      << std::endl;
        }
#if 0
        int dev_mode_num = 0;
        while (EnumDisplaySettings((const char*)disp_device.DeviceName, dev_mode_num, &dev_mode)) {

            std::cout << dev_mode_num << ": dev_mode._width = " << dev_mode.dmPelsWidth
                      << ",\t dev_mode._height = " << dev_mode.dmPelsHeight
                      << ",\t dev_mode._refresh_rate = " << dev_mode.dmDisplayFrequency
                      << ",\t dev_mode._bpp = " << dev_mode.dmBitsPerPel<< std::endl;
            dev_mode_num++;
        }
#endif
#if 1

        HDC               dev_context = NULL;
        dev_context = CreateIC(disp_device.DeviceName, disp_device.DeviceName, 0, NULL);
        //dev_context = CreateDC(TEXT("DISPLAY"), disp_device.DeviceName, 0, NULL);

        if (!dev_context) {
            std::cerr << "error creating device context" << std::endl;
            char* error_msg;

            if (0 == FormatMessage(       FORMAT_MESSAGE_IGNORE_INSERTS
                                        | FORMAT_MESSAGE_FROM_SYSTEM
                                        | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                                        0,
                                        GetLastError(),
                                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                        (LPTSTR)&error_msg,
                                        0,
                                        0)) {
                std::cerr << "FormatMessage failed (" + boost::lexical_cast<std::string>(GetLastError()) << std::endl;
            }
            std::string msg(error_msg);
            LocalFree(error_msg);

            std::cerr << msg << std::endl;
            return (false);
        }

        int disp_width  = ::GetDeviceCaps(dev_context, HORZRES);//GetSystemMetrics(SM_CXVIRTUALSCREEN);//
	    int disp_height = ::GetDeviceCaps(dev_context, VERTRES);

        std::cout << "disp_width    = " << disp_width << std::endl;
        std::cout << "disp_height   = " << disp_height << std::endl;


        // enumerate pixel formats
        scm::gl::test::wgl _wgl;

        if (!_wgl.initialize()) {
            std::cerr << "wgl initialize failed" << std::endl;
            return (false);
        }
        //if (!wgl::is_initialized()) {
        //    if (!wgl::initialize_wgl()) {
        //        return (false);
        //    }
        //}
        int*              pixel_fmts = NULL;
        unsigned int      num_pixel_fmts;


        int query_num_attrib[] = { WGL_NUMBER_PIXEL_FORMATS_ARB };
        int query_num_values[2];

        //gl::output_device_descriptor::device_mode_info_map_t::iterator out_dev_mode_it;
        //for (out_dev_mode_it = desc._device_mode_infos.begin();
        //     out_dev_mode_it != desc._device_mode_infos.end();
        //     out_dev_mode_it++)
        {

            // query max number of pixel formats
            if (!_wgl.wglGetPixelFormatAttribivARB(dev_context, 0, 0, 1, query_num_attrib, query_num_values)) {
                std::cerr << "Warning: wglGetPixelFormatAttribiv returned false for device num "
                                  + boost::lexical_cast<std::string>(i) << std::endl;
                                  //+ std::string("\n\twhen querying max formats for display mode at bpp: "
                                  //+ boost::lexical_cast<std::string>(out_dev_mode_it->first)));
                char* error_msg;

                if (0 == FormatMessage(       FORMAT_MESSAGE_IGNORE_INSERTS
                                            | FORMAT_MESSAGE_FROM_SYSTEM
                                            | FORMAT_MESSAGE_ALLOCATE_BUFFER,
                                            0,
                                            GetLastError(),
                                            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                            (LPTSTR)&error_msg,
                                            0,
                                            0)) {
                    std::cerr << "FormatMessage failed (" + boost::lexical_cast<std::string>(GetLastError()) << std::endl;
                }
                std::string msg(error_msg);
                LocalFree(error_msg);

                std::cerr << msg << std::endl;
                continue;
            }
            num_pixel_fmts = query_num_values[0];
            //pixel_fmts = new int[num_pixel_fmts];

            if (num_pixel_fmts < 1) {
                std::cerr << "Warning: wglGetPixelFormatAttribiv returned 0 pixelformats for devide num"
                                  + boost::lexical_cast<std::string>(i) << std::endl;
                                  //+ std::string("\n\twhen querying max formats for display mode at bpp: "
                                  //+ boost::lexical_cast<std::string>(out_dev_mode_it->first)));
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

            for (unsigned f = 1; f <= num_pixel_fmts /*num_supported_pixel_fmts*/; ++f) {
                //gl::output_device_descriptor::pixel_format_attribs context_attribs;

                if (!_wgl.wglGetPixelFormatAttribivARB(dev_context, f/*pixel_fmts[f]*/, 0, 11, query_fmt_attrib, query_fmt_values)) {
                std::cerr << "Warning: wglGetPixelFormatAttribiv returned false for device num "
                                  + boost::lexical_cast<std::string>(i)
                                  + std::string("\n\twhen querying format attribs for pixel format:")
                                  + boost::lexical_cast<std::string>(f) << std::endl;
                                  //+ std::string("\n\tat display mode bpp: ")
                                  //+ boost::lexical_cast<std::string>(out_dev_mode_it->first));
                    continue;
                }

                if (   /*query_fmt_values[0]  == 32//out_dev_mode_it->first
                    &&*/ query_fmt_values[8]  == WGL_FULL_ACCELERATION_ARB
                    && query_fmt_values[9]  == GL_TRUE
                    && query_fmt_values[10] == WGL_TYPE_RGBA_ARB) {

                    //std::cout << f << ": color_bits = " << query_fmt_values[0]
                    //          << ": depth_bits = " << query_fmt_values[1]
                    //          << ",\t alpha_bits = " << query_fmt_values[2]
                    //          << ",\t stencil_bits = " << query_fmt_values[3]
                    //          << ",\t aux_buffers = " << query_fmt_values[4]
                    //          << ",\t samples = " << query_fmt_values[5]
                    //          << ",\t doublebuffer = " << ((query_fmt_values[6] == GL_TRUE) ? true : false)
                    //          << ",\t stereo = " << ((query_fmt_values[7] == GL_TRUE) ? true : false) << std::endl;
                }
            }
        }


        DeleteDC(dev_context);
#endif
    }

    return (true);
}


int main(int argc, char **argv)
{
    scm::shared_ptr<scm::core>      scm_core(new scm::core(argc, argv));
    std::cout << "sick, sad world..." << std::endl;

    //output_device_info();


    ///////////////////////////////////////////////////////////////////////////////////////////////
    using namespace scm;
    using namespace scm::gl;
    using namespace scm::math;

    wm::display_ptr             _display1(new wm::display("\\\\.\\DISPLAY1"));
    wm::display_ptr             _display2(new wm::display("\\\\.\\DISPLAY2"));

    wm::surface::format_desc    _pixel_format(FORMAT_RGBA_8, FORMAT_D24_S8, true);
    wm::window_ptr              _window1(new wm::window(_display1, "scm::wm test display 1",
                                                        vec2i(0, 0), vec2ui(1920, 1200),
                                                        _pixel_format));
    wm::window_ptr              _window2(new wm::window(_display2, "scm::wm test display 2",
                                                        vec2i(0, 0), vec2ui(1680, 1050),
                                                        _pixel_format));
    wm::context::attribute_desc _context_attribs(4, 1);
    wm::context_ptr             _context1(new wm::context(_window1, _context_attribs));
    wm::context_ptr             _context2(new wm::context(_window2, _context_attribs));

    _context1->make_current(_window1);
    _context2->make_current(_window2);
    //wm::window              _window2(_display2, "scm::wm test display 2", vec2i(0, 0), vec2ui(168, 1050), _pixel_format);

    scm::size_t frame_num = 0;

    _window1->show();
    _window2->show();
    ///////////////////////////////////////////////////////////////////////////////////////////////

    //std::cin.ignore((std::numeric_limits<std::streamsize>::max)(), '\n');
    MSG         tMsg;

    while(1) {
        if (PeekMessage(&tMsg, NULL, 0, 0, PM_REMOVE)) {
            if (tMsg.message ==  WM_LBUTTONDOWN) {
                std::cout << "close" << std::endl;
                break;
            }
            else {
                TranslateMessage( &tMsg );
                DispatchMessage( &tMsg );
            }
            InvalidateRect(_window1->window_handle(), NULL, FALSE);
            InvalidateRect(_window2->window_handle(), NULL, FALSE);

            {
                _context1->make_current(_window1);
                if (frame_num % 2) {
                    glClearColor(1, 0, 0, 0);
                }
                else {
                    glClearColor(0, 0, 1, 0);
                }
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                _window1->swap_buffers(1);
            }

            {
                _context2->make_current(_window2);
                if (frame_num % 2) {
                    glClearColor(1, 0, 0, 0);
                }
                else {
                    glClearColor(0, 0, 1, 0);
                }
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                _window2->swap_buffers(1);
            }

            frame_num++;
        }
    }

//    return tMsg.wParam;

    _window2->hide();
    _window1->hide();

    _context2.reset();
    _context1.reset();
    _window2.reset();
    _window1.reset();

    _display2.reset();
    _display1.reset();

    std::cout << "bye sick, sad world..." << std::endl;

    return (0);
}