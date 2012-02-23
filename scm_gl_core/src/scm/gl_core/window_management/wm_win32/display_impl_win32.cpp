
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "display_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>

#include <boost/lexical_cast.hpp>
#include <boost/unordered_map.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <scm/log.h>
#include <scm/core/memory.h>

#include <scm/gl_core/window_management/wm_win32/util/classic_context_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/error_win32.h>
#include <scm/gl_core/window_management/wm_win32/util/wgl_extensions.h>

namespace scm {
namespace gl {
namespace wm {

namespace detail {

typedef boost::unordered_map<std::string, shared_ptr<display_info> > display_info_map;

bool
enum_display_infos(display_info_map& display_infos, std::ostream& os)
{
    display_infos.clear();

    int num_output_devices = ::GetSystemMetrics(SM_CMONITORS);

    if (num_output_devices < 1) {
        os << "display::display_impl::enum_display_infos() <win32>: " 
           << "unable to determine number of monitor output devices in system";
        return (false);
    }

    for (int i = 0; i < num_output_devices; ++i) {
        DISPLAY_DEVICE  disp_device;
        DEVMODE         disp_mode;

        ::ZeroMemory(&disp_device, sizeof(disp_device));
        disp_device.cb = sizeof(DISPLAY_DEVICE);

        if (0 == ::EnumDisplayDevices(NULL, i, &disp_device, 0)) {
            os << "display::display_impl::enum_display_infos() <win32>: " 
               << "EnumDisplayDevices failed for device num " + boost::lexical_cast<std::string>(i);
            return (false);
        }

        if (0 == ::EnumDisplaySettings(disp_device.DeviceName, ENUM_CURRENT_SETTINGS, &disp_mode)) {
            os << "display::display_impl::enum_display_infos() <win32>: " 
               << "EnumDisplaySettings failed for device num " + boost::lexical_cast<std::string>(i);
            //return (false);
            
        }
        else {
        //std::cout << "dev_name = " << dev_mode.dmDeviceName << ",\tdev_mode._width = " << dev_mode.dmPelsWidth << ",\t dev_mode._height = " << dev_mode.dmPelsHeight
        //          << ",\t dev_mode._refresh_rate = " << dev_mode.dmDisplayFrequency << ",\t dev_mode._bpp = " << dev_mode.dmBitsPerPel << ",\t dev_pos.x = " << dev_mode.dmPosition.x 
        //          << ",\t dev_pos.y = " << dev_mode.dmPosition.y << std::endl;

            shared_ptr<display_info>    di(new display_info());

            di->_dev_name               = disp_device.DeviceName;
            di->_dev_string             = disp_device.DeviceString;
            di->_screen_origin          = math::vec2i(disp_mode.dmPosition.x, disp_mode.dmPosition.y);
            di->_screen_size            = math::vec2i(disp_mode.dmPelsWidth, disp_mode.dmPelsHeight);
            di->_screen_refresh_rate    = disp_mode.dmDisplayFrequency;
            di->_screen_bpp             = disp_mode.dmBitsPerPel;

            display_infos.insert(std::make_pair(di->_dev_name, di));
        }
    }

    return !display_infos.empty();
}

std::string
default_display_name()
{
    DISPLAY_DEVICE  disp_device;

    ::ZeroMemory(&disp_device, sizeof(disp_device));
    disp_device.cb = sizeof(DISPLAY_DEVICE);

    if (0 == ::EnumDisplayDevices(NULL, 0, &disp_device, 0)) {
        err() << log::error
              << "display::display_impl::default_display_name() <win32>: " 
              << "EnumDisplayDevices failed for device num " + boost::lexical_cast<std::string>(0) << log::end;
        return ("");
    }
    else {
        return (std::string(disp_device.DeviceName));
    }
}

} // namespace detail

display::display_impl::display_impl(const std::string& name)
  : _hinstance(0),
    _device_handle(0),
    _wgl_extensions(new util::wgl_extensions())
{
    try {
        _hinstance = ::GetModuleHandle(0);

        if (!_hinstance) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: " 
              << "unable to get module handle "
              << "(system message: " << util::win32_error_message() << ")" << log::end;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));

        }

        std::string class_name(name + boost::lexical_cast<std::string>(boost::uuids::random_generator()()));

        WNDCLASSEX wnd_class;
        ZeroMemory(&wnd_class, sizeof(WNDCLASSEX));
        wnd_class.cbSize        = sizeof(WNDCLASSEX);
        wnd_class.lpfnWndProc   = &DefWindowProc;      
        wnd_class.style         = CS_OWNDC | CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;
        wnd_class.hInstance     = _hinstance;
        wnd_class.hbrBackground = 0;//(HBRUSH)::GetStockObject(DKGRAY_BRUSH);
        wnd_class.lpszClassName = class_name.c_str();

        _window_class = ::RegisterClassEx(&wnd_class);

        if (0 == _window_class) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: " 
              << "unable to register window class (" << class_name << ") "
              << "(system message: " << util::win32_error_message() << ")";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }
        //else {
        //    std::cout << class_name << " registered" << std::endl;
        //}

        detail::display_info_map display_infos;

        std::stringstream dsp_enum_err;
        if (!detail::enum_display_infos(display_infos, dsp_enum_err)) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: " 
              << "unable to enumerate displays: " << dsp_enum_err.str() << std::endl;
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::string disp_name;
        if (name.empty()) {// == std::string("")) {
            disp_name = detail::default_display_name();
        }
        else {
            disp_name = name;
        }

        detail::display_info_map::const_iterator dsp = display_infos.find(disp_name);

        if (dsp == display_infos.end()) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: " 
              << "unable find display (" << disp_name << ")" << std::endl
              << "available displays:" << std::endl;
            for (detail::display_info_map::const_iterator i = display_infos.begin();
                 i != display_infos.end(); ++i) {
                s << i->first << std::endl;
            }
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        _info           = dsp->second;
        _device_handle  = ::CreateDC(TEXT("DISPLAY"), _info->_dev_name.c_str(), 0, NULL);

        if (0 == _device_handle) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: "
              << "unable to create device context "
              << "(system message: " << util::win32_error_message() << ")";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        // setup a dummy gl context to initialize the wgl extensions
        scm::scoped_ptr<util::classic_gl_window>    dummy_window;
        scm::scoped_ptr<util::classic_gl_context>   dummy_context;

        dummy_window.reset(new util::classic_gl_window(_info->_screen_origin, math::vec2ui(10, 10)));
        if (!dummy_window->valid()) {
             std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: "
              << "unable to create dummy window for WGL initialization.";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
       }
        dummy_context.reset(new util::classic_gl_context(*dummy_window));
        if (!dummy_context->valid()) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: "
              << "unable to create dummy window context for WGL initialization.";
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        std::stringstream init_err;
        if (!_wgl_extensions->initialize(init_err)) {
            std::ostringstream s;
            s << "display::display_impl::display_impl() <win32>: "
              << "unable to initialize WGL ARB extensions, WGL init failed: "
              << init_err.str();
            //err() << log::fatal << s.str() << log::end;
            throw(std::runtime_error(s.str()));
        }

        dummy_context.reset();
        dummy_window.reset();
    }
    catch(...) {
        cleanup();
        throw;
    }
}

display::display_impl::~display_impl()
{
    cleanup();
}

void
display::display_impl::cleanup()
{
    if (_device_handle) {
        if (0 == ::DeleteDC(_device_handle)) {
            err() << log::error
                  << "display::display_impl::~display_impl() <win32>: " 
                  << "unable to delete device context "
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
        }
    }
    if (_window_class) {
        if (0 == ::UnregisterClass(reinterpret_cast<LPCSTR>(_window_class), _hinstance)) {
            err() << log::error
                  << "display::display_impl::~display_impl() <win32>: " 
                  << "unable to unregister window class "
                  << "(system message: " << util::win32_error_message() << ")" << log::end;
        }
    }
    _info.reset();
}


} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
