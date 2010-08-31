
#include "display_impl_win32.h"

#if SCM_PLATFORM == SCM_PLATFORM_WINDOWS

#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>

#include <boost/unordered_map.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <scm/log.h>
#include <scm/core/pointer_types.h>

#include <scm/gl_util/window_management/wm_win32/error_win32.h>

namespace scm {
namespace gl {
namespace wm {

namespace detail {

typedef boost::unordered_map<std::string, shared_ptr<display_info> > display_info_map;

bool
enum_display_infos(display_info_map& display_infos)
{
    display_infos.clear();

    int num_output_devices = ::GetSystemMetrics(SM_CMONITORS);

    if (num_output_devices < 1) {
        err() << log::error
              << "display::display_impl::enum_display_infos() <win32>: " 
              << "unable to determine number of monitor output devices in system" << log::end;
        return (false);
    }

    for (int i = 0; i < num_output_devices; ++i) {
        DISPLAY_DEVICE  disp_device;
        DEVMODE         disp_mode;

        ::ZeroMemory(&disp_device, sizeof(disp_device));
        disp_device.cb = sizeof(DISPLAY_DEVICE);

        if (0 == ::EnumDisplayDevices(NULL, i, &disp_device, 0)) {
            err() << log::error
                  << "display::display_impl::enum_display_infos() <win32>: " 
                  << "EnumDisplayDevices failed for device num " + boost::lexical_cast<std::string>(i) << log::end;
            return (false);
        }

        if (0 == ::EnumDisplaySettings(disp_device.DeviceName, ENUM_CURRENT_SETTINGS, &disp_mode)) {
            err() << log::error
                  << "display::display_impl::enum_display_infos() <win32>: " 
                  << "EnumDisplaySettings failed for device num " + boost::lexical_cast<std::string>(i) << log::end;
            return (false);
            
        }
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

    return (true);
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
{
    _hinstance = ::GetModuleHandle(0);

    if (!_hinstance) {
        err() << log::error
              << "display::display_impl::display_impl() <win32>: " 
              << "unable to get module handle"
              << " - system message: " << log::nline
              << util::win32_error_message()
              << log::end;
    }

    std::stringstream class_name;
    class_name << name << boost::uuids::random_generator()();

    WNDCLASSEX wnd_class;
    ZeroMemory(&wnd_class, sizeof(WNDCLASSEX));
    wnd_class.cbSize        = sizeof(WNDCLASSEX);
    wnd_class.style         = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    wnd_class.hInstance     = _hinstance;
    wnd_class.lpszClassName = class_name.str().c_str();

    _window_class = ::RegisterClassEx(&wnd_class);

    if (0 == _window_class) {
        std::ostringstream s;
        s << log::error
          << "display::display_impl::display_impl() <win32>: " 
          << "unable to register window class (" << name << ")"
          << " - system message: " << std::endl
          << util::win32_error_message();
        err() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }

    detail::display_info_map display_infos;

    if (!detail::enum_display_infos(display_infos)) {
        std::ostringstream s;
        s << log::error
          << "display::display_impl::display_impl() <win32>: " 
          << "unable to enumerate displays." << std::endl;
        err() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }

    std::string disp_name;
    if (!name.empty()) {
        disp_name = name;
    }
    else {
        disp_name = detail::default_display_name();
    }

    detail::display_info_map::const_iterator dsp = display_infos.find(name);

    if (dsp == display_infos.end()) {
        std::ostringstream s;
        s << log::error
          << "display::display_impl::display_impl() <win32>: " 
          << "unable find display (" << name << ")" << std::endl
          << "available displays:" << std::endl;
        for (detail::display_info_map::const_iterator i = display_infos.begin();
             i != display_infos.end(); ++i) {
            s << i->first << std::endl;
        }
        err() << log::fatal << s.str() << log::end;
        throw(std::runtime_error(s.str()));
    }

    _info = dsp->second;
}

display::display_impl::~display_impl()
{
    if (0 == ::UnregisterClass(reinterpret_cast<LPCSTR>(_window_class), _hinstance)) {
        err() << log::error
              << "display::display_impl::~display_impl() <win32>: " 
              << "unable to unregister window class"
              << " - system message: " << log::nline
              << util::win32_error_message()
              << log::end;
    }
}

} // namespace wm
} // namepspace gl
} // namepspace scm

#endif // SCM_PLATFORM == SCM_PLATFORM_WINDOWS
